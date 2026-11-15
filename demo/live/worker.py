"""One audit run for BEAT Audit Live, in its own process.

The server starts this file ahead of time so that NumPy, scikit-learn and ``beat_audit`` are
already imported when a file arrives. Protocol on stdout, one JSON object per line:
``{"type": "ready"}``, then, after one job line ``{"path": ..., "seconds_per_output": ...}``
on stdin, ``progress`` events and finally ``result`` or ``error``. Everything else the
libraries print goes to stderr.

The audit itself is ``beat_audit.audit_stream`` with the settings of ``demo_config``. Progress
is observed by wrapping ``beat_audit.engine._analyse`` in this process only; the wrapper passes
arguments and results through unchanged. Its interim AUCs are shown before Holm's correction.
"""

from datetime import datetime, timezone
import json
import os
import sys
import time

PROTOCOL = sys.stdout
sys.stdout = sys.stderr

import numpy as np  # noqa: E402

import beat_audit  # noqa: E402
from beat_audit import AuditConfig, audit_stream  # noqa: E402
from beat_audit import engine  # noqa: E402
from beat_audit.io import file_hash, load_input  # noqa: E402
from beat_audit.protocol import chronological_days  # noqa: E402

# Shortened inference budgets for a live run (the tool labels them nonstandard), and the
# diagnostic analyses the card does not show. Features use the tool's base cosine attacker only:
# with 20 users cosine_rich took two thirds of a ~37 s run, beyond the ~15 s a live run may take.
# Everything else is the tool's default.
DEMO = {"permutations": 2000, "bootstraps": 1000, "gallery": False, "closed_set": False,
        "sequential": False, "threads": 4, "comparators": ("cosine",)}
STANDARD = {"permutations": 10_000, "bootstraps": 2000}
# The tool's standard attack per kind: commands always use the Bayes factor alone.
STANDARD_COMPARATORS = {"features": list(AuditConfig().comparators), "commands": ["bayes_factor"]}
K_CANDIDATES = (1, 5, 10, 25, 50, 100)
HEADLINE = {"features": "cosine", "commands": "bayes_factor"}

# Tool messages are "field: action". Korean text for each field the tool can name.
FIELD_MESSAGES = {
    "outputs": "출력 열을 읽을 수 없습니다. 명령은 0 이상의 정수 하나(outputs), 특징은 outputs_0, "
               "outputs_1, … 숫자 열이어야 합니다.",
    "user": "사용자가 6명 이상이어야 합니다. 사용자를 나눠 공격자를 학습·평가하기 때문입니다.",
    "day": "기록일이 둘 이상이어야 합니다. 같은 날과 다른 날을 비교하기 때문입니다.",
    "day_order": "기록일 라벨의 순서를 알 수 없습니다. 숫자(1, 2, …)나 날짜(2026-11-16)로 적어 주세요.",
    "order": "order 열은 사용자·기록일마다 겹치지 않는 시간 순서여야 합니다.",
    "segment": "segment 열에 빈 값이 있습니다. 모든 행에 구간 라벨을 적어 주세요.",
    "target": "target(과제 단서) 열에 빈 값이 있습니다.",
    "C": "C(명령 종류 수) 열은 모든 행에서 같은 양의 정수이고 가장 큰 명령 코드보다 커야 합니다.",
    "input fields": "알 수 없는 열이 있습니다. 허용되는 열: user, day, segment, order, target, C, "
                    "outputs 또는 outputs_0, outputs_1, …",
    "kind": "명령인지 특징인지 판단할 수 없습니다.",
    "k": "출력 수 k를 정할 수 없습니다. 사용자·기록일·구간마다 행이 충분한지 확인해 주세요.",
}


def korean_message(exc):
    text = str(exc)
    field = text.split(":", 1)[0].strip()
    if field.startswith("input fields"):
        field = "input fields"
    return FIELD_MESSAGES.get(field, "도구가 이 파일을 받아들이지 않았습니다."), text


def k_grid(stream):
    """k values that leave two windows in every segment for at least 90% of user-day-segments."""
    counts = [int(np.count_nonzero((stream.user == u) & (stream.day == d) & (stream.segment == s)))
              for u in stream.users for d, s in stream.segments
              if np.any((stream.user == u) & (stream.day == d))]
    limit = float(np.quantile(counts, 0.1)) / 2
    grid = tuple(k for k in K_CANDIDATES if k <= limit)
    if not grid:
        raise ValueError("k: every user needs at least one output in each segment of each day")
    return grid


def demo_config(stream, seconds_per_output=None):
    return AuditConfig(k_grid=k_grid(stream), cohort="live-demo",
                       seconds_per_output=seconds_per_output, **DEMO)


def _relation(resolved):
    a, b = resolved[0]
    return "same-day" if {d for d, _ in a} == {d for d, _ in b} else "cross-day"


def planned_steps(stream, config):
    days, _ = chronological_days(stream.day, config.day_order)
    conditions = engine.default_conditions(stream, days)
    names = ("bayes_factor",) if stream.kind == "commands" else config.comparators
    per_k = {k: len([n for n in names if not (k == 1 and n.endswith("_rich"))])
             for k in config.k_grid}
    return len(conditions) * sum(per_k.values()), [c.relation for c in conditions]


def headline_comparator(kind, k, comparators):
    """The pre-named comparator: cosine for features, bayes_factor for commands."""
    name = HEADLINE[kind]
    return name if name in comparators else None


def summarize(report, stream, config, *, path, filename, elapsed):
    """The fields the card and the print page show, taken from the tool's report."""
    conditions = []
    for condition in report["configuration"]["conditions"]:
        cells = []
        for cell in sorted((c for c in report["cells"] if c["condition"] == condition["name"]),
                           key=lambda c: c["k"]):
            entry = {"k": cell["k"], "status": cell["status"], "reason": cell.get("reason"),
                     "users_analysed": len(cell["analysed_users"]),
                     "excluded_users": cell["excluded_users"]}
            if cell["status"] == "ok":
                name = headline_comparator(report["kind"], cell["k"], cell["comparators"])
                value = cell["comparators"][name]
                strongest = cell["comparators"][cell["strongest"]]
                entry.update(
                    wording=cell["wording"], comparator=name, auc=value["auc"],
                    interval=value["interval"], p=value["p"], holm_p=value["holm_p"],
                    comparator_wording=value["wording"], unreliable=value["unreliable"],
                    strongest=cell["strongest"], strongest_auc=strongest["auc"],
                    detected_comparators=sorted(n for n, v in cell["comparators"].items()
                                                if v["holm_p"] <= 0.05),
                    comparators=sorted(cell["comparators"]))
            cells.append(entry)
        conditions.append({"name": condition["name"], "relation": condition["relation"],
                           "days": sorted({d for c in report["cells"]
                                           if c["condition"] == condition["name"]
                                           for d in c["days"]}),
                           "cells": cells})
    contrasts = []
    for row in report["contrasts"]:
        if row.get("status") != "ok":
            continue
        name = headline_comparator(report["kind"], row["k"], row["comparators"])
        if name:
            value = row["comparators"][name]
            contrasts.append({"k": row["k"], "comparator": name, "mean": value["mean"],
                              "interval": value["interval"]})
    dimension = int(stream.outputs.shape[1]) if stream.kind == "features" else None
    return {
        "filename": filename,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "tool_runtime_seconds": report["runtime_seconds"],
        "input": {
            "kind": stream.kind, "rows": int(len(stream.user)), "users": len(stream.users),
            "days": report["days"], "day_order": report["day_order"], "dimension": dimension,
            "alphabet": int(stream.alphabet) if stream.kind == "commands" else None,
            "cues": bool(stream.target_given),
            "cue_levels": sorted(set(stream.target.tolist())) if stream.target_given else [],
            "sha256": file_hash(path),
        },
        "settings": {
            "k_grid": list(config.k_grid), "permutations": config.permutations,
            "bootstraps": config.bootstraps,
            "standard": {**STANDARD, "comparators": STANDARD_COMPARATORS[stream.kind]},
            "comparators": (["bayes_factor"] if stream.kind == "commands"
                            else list(config.comparators)),
            "gallery": config.gallery, "closed_set": config.closed_set,
            "sequential": config.sequential, "threads": config.threads,
            "seed": config.seed, "fold_seed": config.fold_seed,
            "seconds_per_output": config.seconds_per_output,
        },
        "conditions": conditions,
        "contrasts": contrasts,
        "smallest_detected_k": report["smallest_detected_k"],
        "warnings": report["warnings"],
        "interpretation": report["interpretation"],
        "hashes": {"source": report["hashes"]["source"]["sha256"],
                   "configuration": report["hashes"]["configuration"],
                   "input": report["hashes"]["inputs"]["0"]},
        "tool_version": beat_audit.__version__,
    }


def run(path, filename, seconds_per_output, emit):
    started = time.perf_counter()
    emit({"type": "progress", "phase": "reading", "done": 0, "total": 0})
    stream = load_input(path)
    config = demo_config(stream, seconds_per_output)
    total, relations = planned_steps(stream, config)
    done = 0
    emit({"type": "progress", "phase": "plan", "done": 0, "total": total,
          "k_grid": list(config.k_grid), "relations": relations, "kind": stream.kind,
          "users": len(stream.users), "rows": int(len(stream.user)),
          "days": sorted(set(stream.day.tolist()))})
    original = engine._analyse

    def observed(streams, blocks, users, groups, resolved, name, cfg, cues, k, **kw):
        nonlocal done
        relation = _relation(resolved)
        emit({"type": "progress", "phase": "step", "done": done, "total": total, "k": k,
              "relation": relation, "comparator": name})
        result = original(streams, blocks, users, groups, resolved, name, cfg, cues, k, **kw)
        done += 1
        emit({"type": "progress", "phase": "point", "done": done, "total": total, "k": k,
              "relation": relation, "comparator": name, "auc": result["auc"],
              "interval": result["interval"]})
        return result

    engine._analyse = observed
    try:
        report = audit_stream(path, config=config)
    finally:
        engine._analyse = original
    failed = [c for c in report["cells"] if c["status"] == "failed"]
    if failed:
        raise RuntimeError(failed[0]["reason"])
    return summarize(report, stream, config, path=path, filename=filename,
                     elapsed=time.perf_counter() - started)


def _emit(event):
    PROTOCOL.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    PROTOCOL.flush()


def main():
    _emit({"type": "ready", "pid": os.getpid()})
    line = sys.stdin.readline()
    if not line:
        return
    job = json.loads(line)
    try:
        summary = run(job["path"], job.get("filename", ""), job.get("seconds_per_output"), _emit)
        _emit({"type": "result", "summary": summary})
    except ValueError as exc:
        message, detail = korean_message(exc)
        _emit({"type": "error", "message": message, "detail": detail})
    except Exception as exc:  # computation failures are reported, never shown as a traceback
        _emit({"type": "error", "message": "감사 계산이 실패했습니다. 저장된 실행이 있으면 그것을 보여 줄 수 있습니다.",
               "detail": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
