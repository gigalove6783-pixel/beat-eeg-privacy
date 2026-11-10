"""Export compact, licence-safe JSON for the static demo in demo/.

Reads result tables (results/) and per-trial replay/attack artifacts (artifacts/exp_c/)
and writes demo/data/. Only aggregated results, decoded commands (left/right/hold),
trial labels and attacker identity predictions are written: no EEG, covariances,
features, representations or logits (BNCI2014-001 is CC BY-ND 4.0).

Story data (all aggregate): experiment status and evidence level (report §5), EEGNet
identifier vs independent attackers (results/exp_e/lambda_summary.csv), the experiment-G
enrolment, scenario and reproduction tables, the audit checklist (report §9), the
pre-registration timeline (docs/decision_log.md), the Lee2019 replication and experiment K's
randomized-response vs equalized-decoding frontier (results/exp_k/frontier.csv). The
replication stays "pending" until results/exp_l/decision.json exists; re-running this
script then fills in the claim decisions without any change to the page.

Usage:  .venv\\Scripts\\python scripts\\export_demo_data.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beat.exp_k import make_groups  # noqa: E402
RESULTS, ARTIFACTS = ROOT / "results", ROOT / "artifacts"
OUT = ROOT / "demo" / "data"
EPISODE = 20
CMD = {-1: "L", 0: "H", 1: "R"}
KS = (1, 5, 10, 20)
E_HALVES = (((1, 2, 3), (4, 5, 6)), ((4, 5, 6), (1, 2, 3)))  # beat/splits.py
# Every experiment of the study with the result file whose presence marks it as done.
# Evidence level and outcome text come from the status table in docs/report.md §5.
EXPERIMENTS = {
    "exp_a": ("A: decoding baseline", "exp_a/h1.json"),
    "exp_b": ("B: pooled models, identity adversary", "exp_b/tradeoff.csv"),
    "exp_c": ("C: replayed arm, observation attack", "exp_c/observation_strongest.csv"),
    "exp_d": ("D: bottleneck sweep, stronger adversaries", "exp_d/tradeoff.csv"),
    "exp_e": ("E: EEGNet identifier vs independent attackers", "exp_e/lambda_summary.csv"),
    "exp_g": ("G: published reference-session defence", "exp_g/bnci/enrolment_averaged.csv"),
    "exp_l": ("L: Lee2019/OpenBMI replication, 54 users", "exp_l/decision.json"),
    "exp_k": ("K: command streams; randomized response vs equalized decoding", "exp_k/frontier.csv"),
}
# Result-table scenario names as labelled in README.md.
SCENARIO = {"main": "within-day", "auxiliary": "cross-day"}
GENERIC_EXPERIMENTS = ("exp_b", "exp_c", "exp_d", "exp_e")
GENERIC_TABLES = {"tradeoff": "tradeoff.csv", "adversary": "adversary_comparison.csv",
                  "observation": "observation_strongest.csv"}
# Columns that would carry per-trial signal content; refuse to export them.
FORBIDDEN = re.compile(r"representation|feature_\d|logit|cov|eeg|^x\d", re.I)

sources: dict[str, str] = {}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def track(path: Path) -> Path:
    sources[rel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(track(path))


def read_json(path: Path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else round(float(value), 6)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def records(frame: pd.DataFrame, **extra) -> list[dict]:
    bad = [c for c in frame.columns if FORBIDDEN.search(str(c))]
    if bad:
        raise ValueError(f"Refusing to export per-trial signal columns: {bad}")
    return [dict(extra, **{k: clean(v) for k, v in row.items()})
            for row in frame.to_dict(orient="records")]


def src(path: str, **where) -> dict:
    return dict(file=path, **({"where": where} if where else {}))


# ---------------------------------------------------------------- generic tables

def generic():
    data = {name: [] for name in GENERIC_TABLES}
    for exp in GENERIC_EXPERIMENTS:
        for name, file in GENERIC_TABLES.items():
            path = RESULTS / exp / file
            if path.exists():
                frame = read_csv(path)
                if name == "observation":
                    frame = frame[[c for c in frame.columns if not c.startswith("macro_f1")]]
                data[name].extend(records(frame, experiment=exp))
    return data


# ---------------------------------------------------------------- report text

REPORT = "docs/report.md"
TEX = {r"\lambda": "λ", r"\ge": "≥", r"\le": "≤", r"\ldots": "…", r"\approx": "≈",
       r"\times": "×", r"\in": "∈", r"\{": "{", r"\}": "}", r"\text": "", "~": " "}


def plain(text: str) -> str:
    """Markdown/LaTeX fragment of the report as plain text."""
    def math(m):
        s = m.group(1)
        for k, v in TEX.items():
            s = s.replace(k, v)
        return re.sub(r"[{}]", "", s)
    text = re.sub(r"\$([^$]+)\$", math, text)
    return re.sub(r"\*\*|\*|`", "", text).strip()


def report_section(number: str) -> str:
    text = track(ROOT / REPORT).read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(number)} .*?$(.*?)(?=^## )", text, re.M | re.S)
    if not match:
        raise ValueError(f"Section {number} not found in {REPORT}")
    return match.group(1)


def experiments():
    rows = [l for l in report_section("5").splitlines() if l.startswith("| ") and "---" not in l]
    table = {}
    for line in rows[1:]:
        letter, question, status, outcome = [plain(c) for c in line.strip("|").split("|")]
        table[f"exp_{letter.lower()}"] = dict(question=question, status_text=status,
                                              outcome=outcome)
    out = {}
    for exp, (title, marker) in EXPERIMENTS.items():
        row = table[exp]
        # README's three tiers: a confirmatory but descriptive experiment (C) fixed its analysis in
        # a sealed plan without a decision rule, i.e. a pre-registered estimate.
        status = row["status_text"].lower()
        evidence = "estimate" if status.startswith("confirmatory (descriptive)") else status.split()[0]
        if evidence not in ("confirmatory", "estimate", "exploratory"):
            raise ValueError(f"Unknown evidence level for {exp}: {row['status_text']}")
        done = (RESULTS / marker).exists()
        out[exp] = dict(title=title, letter=exp[-1].upper(), evidence=evidence,
                        status="available" if done else "pending",
                        question=row["question"], status_text=row["status_text"],
                        # The report's outcome cell for a pending experiment is a placeholder.
                        outcome=row["outcome"] if done else None,
                        source=src(f"results/{marker}" if done else REPORT,
                                   **({} if done else {"section": "5"})),
                        outcome_source=src(REPORT, section="5"))
    # Model seeds behind the seed-mean values shown for B, D and E.
    cfg = lambda e: yaml.safe_load(track(ROOT / "configs" / f"{e}.yaml").read_text(encoding="utf-8"))
    b, d, e = cfg("exp_b"), cfg("exp_d"), cfg("exp_e")
    if "seeds" not in d["inherit_from_exp_b"]:
        raise ValueError("configs/exp_d.yaml no longer inherits seeds from exp_b")
    out["exp_b"]["seeds"] = dict(n=len(b["seeds"]), source=src("configs/exp_b.yaml", key="seeds"))
    out["exp_d"]["seeds"] = dict(n=len(b["seeds"]), source=src("configs/exp_d.yaml", key="inherit_from_exp_b",
                                                                 inherited="configs/exp_b.yaml seeds"))
    out["exp_e"]["seeds"] = dict(n=len(e["seeds"]), source=src("configs/exp_e.yaml", key="seeds"))
    return out


def checklist():
    items = []
    for m in re.finditer(r"^(\d+)\. \*\*(.+?)\*\* (.+)$", report_section("9"), re.M):
        items.append(dict(n=int(m.group(1)), title=plain(m.group(2)).rstrip("."),
                          text=plain(m.group(3))))
    if len(items) < 5:
        raise ValueError("Audit checklist not found in docs/report.md §9")
    return dict(items=items, source=src(REPORT, section="9"))


# ---------------------------------------------------------------- pitfalls 1 and 2

def eegnet(chance):
    frame = read_csv(RESULTS / "exp_e" / "lambda_summary.csv")
    near = frame[frame.identifier_ba - chance <= 0.08]
    where = dict(rule="identifier_ba - chance <= 0.08")
    return dict(rows=records(frame, experiment="exp_e"), source="results/exp_e/lambda_summary.csv",
                near_chance=dict(identifier=[clean(near.identifier_ba.min()),
                                             clean(near.identifier_ba.max())],
                                 independent=[clean(near.strongest_ba.min()),
                                              clean(near.strongest_ba.max())],
                                 source=src("results/exp_e/lambda_summary.csv", **where)))


def pitfall2():
    base = RESULTS / "exp_g" / "bnci"
    enrol = read_csv(base / "enrolment_averaged.csv")
    enrol = enrol[["condition", "attacker", "enrolment_n", "k", "accuracy", "status", "reason",
                   "directions_evaluated"]]
    ident = read_csv(base / "identity_averaged.csv")[["condition", "scenario", "attacker", "k",
                                                      "accuracy"]]
    paper = read_csv(base / "paper_table1_comparison.csv")
    paper = paper[["condition", "direction", "attacker", "paper_accuracy_percent",
                   "observed_accuracy_percent", "comparable", "difference_pp"]]
    e = enrol.assign(enrolment_n=enrol.enrolment_n.astype(str))
    cell = lambda c, a, n: e[(e.condition == c) & (e.attacker == a) & (e.enrolment_n == n)
                             & (e.k == 1)].accuracy.iloc[0]
    rep = lambda c, d: paper[(paper.condition == c) & (paper.direction == d)
                             & (paper.attacker == "paper_ts_svm")].iloc[0]
    k1 = ident[ident.k == 1]
    span = lambda f: [clean(f.accuracy.min()), clean(f.accuracy.max())]
    fe, fi, fp = ("results/exp_g/bnci/enrolment_averaged.csv",
                  "results/exp_g/bnci/identity_averaged.csv",
                  "results/exp_g/bnci/paper_table1_comparison.csv")
    headline = {
        "g_enrol_n1": dict(value=clean(cell("defended", "beat_logistic", "1")),
                           source=src(fe, condition="defended", attacker="beat_logistic",
                                      enrolment_n=1, k=1)),
        "g_enrol_n5": dict(value=clean(cell("defended", "beat_logistic", "5")),
                           source=src(fe, condition="defended", attacker="beat_logistic",
                                      enrolment_n=5, k=1)),
        "g_enrol_n1_undefended": dict(value=clean(cell("undefended", "beat_logistic", "1")),
                                      source=src(fe, condition="undefended",
                                                 attacker="beat_logistic", enrolment_n=1, k=1)),
        "g_repro": dict(value={"paper_undefended": clean(rep("undefended", "S1_to_S2").paper_accuracy_percent),
                               "ours_undefended": clean(rep("undefended", "S1_to_S2").observed_accuracy_percent),
                               "paper_defended": clean(rep("defended", "S1_to_S2").paper_accuracy_percent),
                               "ours_defended": clean(rep("defended", "S1_to_S2").observed_accuracy_percent)},
                        source=src(fp, attacker="paper_ts_svm", direction="S1_to_S2")),
        "g_reference_defended": dict(
            value=span(k1[(k1.condition == "defended") & (k1.scenario == "reference_session")]),
            source=src(fi, condition="defended", scenario="reference_session", k=1)),
        "g_reference_undefended": dict(
            value=span(k1[(k1.condition == "undefended") & (k1.scenario == "reference_session")]),
            source=src(fi, condition="undefended", scenario="reference_session", k=1)),
        "g_post_defended": dict(
            value=span(k1[(k1.condition == "defended") & (k1.scenario == "post_deployment")]),
            source=src(fi, condition="defended", scenario="post_deployment", k=1)),
    }
    data = dict(enrolment=dict(rows=records(e, experiment="exp_g"), source=fe),
                scenarios=dict(rows=records(ident, experiment="exp_g"), source=fi),
                reproduction=dict(rows=records(paper, experiment="exp_g"), source=fp))
    return data, headline


# ---------------------------------------------------------------- timeline and replication

LOG = "docs/decision_log.md"


def log_entry(text: str, pattern: str):
    for line in text.splitlines():
        if line.startswith("| ") and (m := re.search(pattern, line)):
            return line.split("|")[1].strip(), m
    raise ValueError(f"Pattern {pattern!r} not found in {LOG}")


def timeline(lee_status):
    log = track(ROOT / LOG).read_text(encoding="utf-8")
    d1, s1 = log_entry(log, r"\*\*E seal\*\*: commit `([0-9a-f]{40})`.*?All (\d+) tests passed")
    d2, ab = log_entry(log, r"A and B final runs completed at commit `([0-9a-f]{7,40})`")
    d3, s2 = log_entry(log, r"\*\*E seal 2\*\*: commit `([0-9a-f]{40})`.*?(\d+) tests passed")
    d4, s3 = log_entry(log, r"Seal 3 commit: `([0-9a-f]{40})`")
    _, s4 = log_entry(log, r"Seal 4: commit `([0-9a-f]{40})`.*?token time (\d{4}-\d\d-\d\d) "
                           r"(\d\d:\d\d:\d\d) GMT")
    return dict(source=src(LOG), events=[
        dict(kind="seal", label="Seal 1", date=d1, commit=s1.group(1), tests=int(s1.group(2)),
             title="Plan v1 sealed", evidence="confirmatory", experiments=["exp_a", "exp_b", "exp_c"],
             text="Plan, configs, code and session-T choices for A–C fixed before session E is scored."),
        dict(kind="run", label="Session E scored", date=d2, commit=ab.group(1),
             title="A and B final runs", evidence="confirmatory", experiments=["exp_a", "exp_b", "exp_c"],
             text="Held-out session E evaluated at the sealed settings; C follows on the same models."),
        dict(kind="seal", label="Seal 2", date=d3, commit=s2.group(1), tests=int(s2.group(2)),
             title="Plan v2 sealed", evidence="exploratory", experiments=["exp_d", "exp_e"],
             text="Follow-ups D and E fixed before their runs; exploratory because session E was reused."),
        dict(kind="run", label="No seal", date=None, commit=None, title="Experiment G",
             evidence="exploratory", experiments=["exp_g"],
             text="Published defence at its published operating point; no selection on session 2."),
        dict(kind="seal", label="Seal 3", date=d4, commit=s3.group(1), title="Plan v3 sealed",
             evidence="confirmatory", experiments=["exp_l"],
             text="Lee2019 claims C1–C4 fixed with session 2 unopened."),
        dict(kind="pending" if lee_status == "pending" else "run", label="Replication",
             date=None, commit=None, title="Lee2019 final run", evidence="confirmatory",
             experiments=["exp_l"],
             text="Claims C1–C4 decided by Holm's procedure on 54 users."),
        dict(kind="seal", label="Seal 4", date=f"{s4.group(2)} {s4.group(3)} UTC (RFC 3161)",
             commit=s4.group(1), title="Study M sealed and time-stamped", evidence="confirmatory",
             experiments=["exp_m"],
             text="Cross-day claims M1–M3 on 51 new users (Yang2025) fixed; an RFC 3161 token from "
                  "FreeTSA time-stamps the seal commit. The final run is in progress and no result "
                  "is reported yet."),
    ])


# Plan-v3 claims (docs/report.md §6.7; thresholds as coded in beat/exp_l.py infer_claims).
CLAIMS = {
    "C1": "Pitfall 1 at scale: for the adversarially trained model (λ = 1, d = 8) the strongest "
          "independent attacker exceeds the co-trained head by at least 20 pp.",
    "C2": "The recentred task-only bottleneck (d = 8) lowers strongest identity BA by at least "
          "20 pp versus raw features, with task BA non-inferior at a −5 pp margin.",
    "C3": "Commands leak: recentred-bottleneck command identity at τ = 0.5 and k = 20 exceeds "
          "twice chance (2/{n}) and exceeds its k = 1 value.",
    "C4": "Pitfall 2 at scale: the published defence, attacked with five labelled deployment trials "
          "per participant, exceeds 10/{n}, if its reference-session benchmark is reproduced "
          "(within ±5 pp of 2.4%).",
}
CLAIM_FIELDS = ("decision", "p", "holm_p", "observed_difference", "effect_requirement",
                "benchmark_gate")


LD = "results/exp_l/decision.json"
LI = "results/exp_l/identity_summary.csv"
LE = "results/exp_l/enrolment.csv"
BOOT = "95% participant bootstrap"
TLOW = "one-sided 95% paired-t lower bound"


def quantity(qid, label, value, fmt, source, ci=None, ci_kind=None, seeds=None, note=None):
    return dict(id=qid, label=label, value=clean(value), fmt=fmt, source=source,
                ci=None if ci is None else [clean(ci[0]), clean(ci[1])], ci_kind=ci_kind,
                seeds=seeds, note=note)


def seed_count(row):
    return len(str(row["seeds"]).split(","))


def lee_quantities(decision, n):
    """Tested quantity of each claim (beat/exp_l.py infer_claims), aggregate only."""
    ident = read_csv(RESULTS / "exp_l" / "identity_summary.csv")
    enrol = read_csv(RESULTS / "exp_l" / "enrolment.csv")
    task = read_csv(RESULTS / "exp_l" / "task_summary.csv").set_index("key")
    repro = read_json(RESULTS / "exp_l" / "reproduction.json")
    claims = decision["claims"]

    def strongest(frame, **where):
        sel = frame[frame.strongest.astype(bool)]
        for k, v in where.items():
            sel = sel[sel[k] == v]
        if len(sel) != 1:
            raise ValueError(f"Expected one strongest row for {where}, got {len(sel)}")
        return sel.iloc[0]

    def ident_q(qid, label, row, where):
        return quantity(qid, f"{label} ({row.attacker}, {SCENARIO[row.scenario]})", row.ba, "ba3",
                        src(LI, **where, strongest=True), ci=(row.ci_low, row.ci_high),
                        ci_kind=BOOT, seeds=seed_count(row))

    # C1: gap = strongest independent mean - head mean, tested per attacker against the head.
    w1 = dict(kind="representation", key="P3_lambda1")
    s1 = strongest(ident, **w1)
    t1 = claims["C1"]["component_tests"][f"{s1.attacker}/{s1.scenario}"]
    gap = claims["C1"]["observed_difference"]
    if abs(t1["mean"] - gap) > 1e-9 or abs(s1.ba - s1.strongest_ba) > 1e-9:
        raise ValueError("C1 gap does not match the strongest attacker's paired test")
    c1 = [quantity("head", "Co-trained identity head", s1.ba - gap, "ba3",
                   src(LD, claim="C1", value="strongest independent − observed_difference"),
                   seeds=seed_count(s1),
                   note="Mean of the head's per-participant recall; no interval is exported."),
          ident_q("independent", "Strongest independent attacker", s1, w1),
          quantity("gap", "Gap", gap, "pp1", src(LD, claim="C1"),
                   ci=(t1["lower"], None), ci_kind=TLOW)]

    # C2: raw features vs recentred d = 8 bottleneck; task non-inferiority at -5 pp.
    w0, w2 = dict(kind="representation", key="P0"), dict(kind="representation", key="P2-RC_d8")
    s0, s2 = strongest(ident, **w0), strongest(ident, **w2)
    u = claims["C2"]["utility"]
    c2 = [ident_q("raw", "Raw features, identity", s0, w0),
          ident_q("bottleneck", "Recentred bottleneck d = 8, identity", s2, w2),
          quantity("task_raw", "Raw features, left/right accuracy", task.loc["P0", "mean"], "ba3",
                   src("results/exp_l/task_summary.csv", key="P0")),
          quantity("task_bottleneck", "Bottleneck, left/right accuracy", task.loc["P2-RC_d8", "mean"],
                   "ba3", src("results/exp_l/task_summary.csv", key="P2-RC_d8")),
          quantity("utility", "Utility change", u["mean"], "pp1", src(LD, claim="C2", field="utility"),
                   ci=(u["lower"], None), ci_kind=TLOW)]

    # C3: recentred-bottleneck commands at tau = 0.5, k = 20 vs twice chance.
    w3 = dict(kind="observation", key="P2-RC_d8", mode="command", tau=0.5)
    s20, s1k = strongest(ident, **w3, k=20), strongest(ident, **w3, k=1)
    c3 = [ident_q("k20", "Commands, k = 20", s20, dict(w3, k=20)),
          ident_q("k1", "Commands, k = 1", s1k, dict(w3, k=1)),
          quantity("twice_chance", "Twice chance", 2 / n, "ba3",
                   src("configs/exp_l.yaml", key="subjects", rule="2 / n"))]

    # C4: published defence, benchmark gate and labelled deployment trials.
    c4 = [quantity("benchmark", "Reference-session benchmark (reproduced)", repro["accuracy"],
                   "pct2", src("results/exp_l/reproduction.json", field="accuracy")),
          quantity("paper", "Published benchmark", repro["paper_accuracy"], "pct1",
                   src("results/exp_l/reproduction.json", field="paper_accuracy"))]
    # 1 and 2 labels: three-seed descriptive curve (enrolment.csv); 5 labels is the tested
    # five-seed cell in identity_summary.csv.
    for m, frame, file, note in (("1", enrol, LE, "Descriptive."), ("2", enrol, LE, "Descriptive."),
                                 ("5", ident, LI, "Tested value.")):
        w = dict(kind="enrolment", key="G", enrolment=m)
        row = strongest(frame.astype({"enrolment": str}), **w)
        c4.append(quantity(f"enrol{m}", f"{m} labelled trial{'s' if m != '1' else ''} per user "
                           f"({row.attacker}, {SCENARIO[row.scenario]})", row.ba, "pct1",
                           src(file, **w, strongest=True),
                           ci=(row.ci_low, row.ci_high), ci_kind=BOOT, seeds=seed_count(row),
                           note=note))
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4}


def lee_headline(claims):
    """Hero numbers for the replication; values come from the claim quantities."""
    q = {c["id"]: {x["id"]: x for x in c["quantities"] or []} for c in claims}
    keys = {"l_raw": ("C2", "raw"), "l_bottleneck": ("C2", "bottleneck"), "l_head": ("C1", "head"),
            "l_independent": ("C1", "independent"), "l_cmd_k20": ("C3", "k20"),
            "l_twice_chance": ("C3", "twice_chance"), "l_benchmark": ("C4", "benchmark"),
            "l_enrol5": ("C4", "enrol5"), "l_utility": ("C2", "utility")}
    out = {}
    for k, (c, qid) in keys.items():
        x = q[c].get(qid)
        out[k] = dict(value=x["value"] if x else None, source=x["source"] if x else src(LD))
    holm = {c["id"]: c["result"]["holm_p"] if c["result"] else None for c in claims}
    out["l_c3_holm"] = dict(value=holm["C3"], source=src(LD, claim="C3", field="holm_p"))
    return out


def lee_disclosures(decision, config):
    """Sealed-rule disclosures for L: MLP non-convergence and the adapted C4 benchmark."""
    report = track(ROOT / REPORT).read_text(encoding="utf-8")
    log = track(ROOT / "docs" / "decision_log.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*MLP non-convergence\.\*\* (\d+) of (\d+) MLP attacker fits, (\d+) of the (\d+) "
                  r"that enter confirmatory tests, reached the fixed (\d+)-iteration budget", report)
    a = re.search(r"(\d+) of (\d+) MLP attacker fits \((\d+) of (\d+) confirmatory\)", log)
    if not m or not a or m.groups()[:4] != a.groups():
        raise ValueError("MLP non-convergence counts missing or inconsistent between report and decision log")
    strongest = re.search(r"All 15 representation-level MLP fits for P3 at \$\\lambda = 1\$ and all 15 for "
                          r"P2-RC at \$d = 8\$ are among them, and the MLP is the strongest attacker for both",
                          report)
    if not strongest:
        raise ValueError("Strongest-MLP non-convergence statement not found in the report")
    adapted = re.search(r"^- \*\*C4 is an adapted replication\.\*\* (.+)$", report, re.M)
    if not adapted:
        raise ValueError("C4 adaptation caveat not found in the report")
    failed, total, conf_failed, conf_total, budget = map(int, m.groups())
    return dict(
        nonconvergence=dict(
            failed=failed, total=total, confirmatory_failed=conf_failed, confirmatory_total=conf_total,
            budget=budget, strongest_c1_c2=True, rule=config["failure_rules"]["nonconvergence"],
            source=[src(REPORT, section="6.7", paragraph="MLP non-convergence"),
                    src("docs/decision_log.md", entry="Independent audit of L results"),
                    src("configs/exp_l.yaml", key="failure_rules.nonconvergence")]),
        c4_caveat=dict(
            reproduction_caveat=decision["reproduction_caveat"], text=plain(adapted.group(1)),
            source=[src(LD, field="reproduction_caveat"),
                    src(REPORT, section="8", item="C4 is an adapted replication")]))


def lee2019(seal3):
    config = yaml.safe_load(track(ROOT / "configs" / "exp_l.yaml").read_text(encoding="utf-8"))
    n = len(config["subjects"])
    path = RESULTS / "exp_l" / "decision.json"
    decision = read_json(path) if path.exists() else None
    quantities = lee_quantities(decision, n) if decision else {}
    claims = []
    for key, text in CLAIMS.items():
        result = None
        if decision:
            c = decision["claims"][key]
            # p-values keep significant digits; clean()'s fixed 6 decimals would turn 1e-26 into 0.
            result = {f: float(f"{c[f]:.6g}") if f in ("p", "holm_p") else clean(c[f])
                      for f in CLAIM_FIELDS if f in c}
            if c.get("reproduction"):
                result["reproduction_accuracy"] = clean(c["reproduction"].get("accuracy"))
        claims.append(dict(id=key, statement=text.format(n=n), result=result,
                           quantities=quantities.get(key)))
    return dict(
        status="available" if decision else "pending", evidence="confirmatory", users=n,
        chance=clean(1 / n), alpha=config["statistics"]["alpha"], seal_commit=seal3,
        claims=claims, disclosures=lee_disclosures(decision, config) if decision else None,
        headline=dict(lee_headline(claims), l_chance=dict(
            value=clean(1 / n), source=src("configs/exp_l.yaml", key="subjects", rule="1 / n"))),
        source=src("results/exp_l/decision.json") if decision else src(REPORT, section="6.7"),
        definition_source=[src(REPORT, section="6.7"), src("configs/exp_l.yaml", key="subjects")],
    )


# ---------------------------------------------------------------- experiment K frontier

KF, KC = "results/exp_k/frontier.csv", "results/exp_k/identity_curves.csv"
# K's declared headline setting (configs/exp_k.yaml decision_rule), as in
# scripts/make_summary_figures.py K_HEAD: Lee2019, P2-RC_d8, tau = 0.8, k = 20, g* = 3.
K_HEAD = dict(dataset="lee2019", condition="P2-RC_d8", tau=0.8, k=20, g=3)
K_GROUPS = (2, 3, 5)
K_RR = ("rr_q0.1", "rr_q0.25", "rr_q0.5", "rr_q0.75")
K_DAYS = ("within_day", "cross_day")


def k_frontier():
    """Aggregate RR vs EQ frontier of K's headline cell, the values of fig9_k_rr_eq_frontier."""
    h, k = K_HEAD, K_HEAD["k"]
    f = read_csv(ROOT / KF)
    f = f[(f.dataset == h["dataset"]) & (f.condition == h["condition"]) & (f.tau == h["tau"])]
    defences = ["none", *K_RR, *(f"{v}_g{g}" for v in ("eqc", "eqm") for g in K_GROUPS),
                "eqc_gall"]
    cols = ["defence", "wrong_weight", "weighted_utility", "weighted_utility_loss", "hold_rate",
            *(f"nway_ba_{d}_k{k}" for d in K_DAYS),
            *(f"g{g}_excess_{d}_k{k}" for d in K_DAYS for g in K_GROUPS)]
    f = f[f.defence.isin(defences)].set_index("defence").loc[defences].reset_index()[cols]
    # N-way users, chance, seeds and attacker count come from identity_curves.csv (frontier.csv
    # has none of them); its strongest-Bayes rows must give frontier.csv's N-way values.
    c = pd.read_csv(track(ROOT / KC), usecols=["dataset", "condition", "tau", "defence", "scenario",
                                                "family_set", "metric", "k", "ba", "chance",
                                                "users", "seeds", "candidates"])
    c = c[(c.dataset == h["dataset"]) & (c.condition == h["condition"]) & (c.tau == h["tau"])
          & (c.metric == "nway") & (c.family_set == "bayes") & (c.k == k)
          & c.scenario.isin(K_DAYS) & c.defence.isin(defences)]
    nway = f.set_index("defence")
    if len(c) != len(defences) * len(K_DAYS) or any(
            abs(r.ba - nway.loc[r.defence, f"nway_ba_{r.scenario}_k{k}"]) > 1e-9
            for r in c.itertuples()):
        raise ValueError("K N-way values differ between frontier.csv and identity_curves.csv")
    meta = {m: sorted({clean(v) for v in c[m]}) for m in ("users", "chance", "seeds", "candidates")}
    if any(len(v) != 1 for v in meta.values()) or len(f.wrong_weight.unique()) != 1:
        raise ValueError("K headline cell mixes user counts, chance, seeds or utility weights")
    where = dict(dataset=h["dataset"], condition=h["condition"], tau=h["tau"])
    # Group sizes from K's own grouping rule (leftover users join the last group); sizes do not
    # depend on the utility ordering, so constant utilities give them.
    track(ROOT / "beat" / "exp_k.py")
    n = int(meta["users"][0])
    sizes = {str(g): [int(s) for s in np.bincount(make_groups(np.arange(n), np.zeros(n), g))]
             for g in K_GROUPS}
    return dict(
        evidence="exploratory", setting=dict(h, enrolment="cross-fitted",
                                             wrong_weight=clean(f.wrong_weight.iloc[0])),
        groups=list(K_GROUPS), group_sizes=sizes,
        group_sizes_source=src("beat/exp_k.py", function="make_groups"),
        rr=list(K_RR), days=list(K_DAYS),
        users=clean(meta["users"][0]), chance=clean(meta["chance"][0]),
        seeds=clean(meta["seeds"][0]), attackers=clean(meta["candidates"][0]),
        rows=records(f, experiment="exp_k"),
        source=src(KF, **where, k=k), chance_source=src(KC, **where, metric="nway",
                                                      family_set="bayes", k=k),
        figure="results/figures/summary_ko/fig9_k_rr_eq_frontier.png")


# ---------------------------------------------------------------- headline numbers

def headline(tradeoff, adversary, observation, chance):
    t = pd.DataFrame([r for r in tradeoff if r["experiment"] == "exp_b"])
    a = pd.DataFrame([r for r in adversary if r["experiment"] == "exp_b"])
    o = pd.DataFrame([r for r in observation if r["experiment"] == "exp_c"])
    cell = lambda c: t[(t.condition == c)].iloc[0]
    ctrl = read_csv(RESULTS / "exp_c" / "control_summary.csv")
    ctrl = ctrl[ctrl["mode"] == "command"]
    thr = read_csv(RESULTS / "exp_c" / "observation_threshold.csv")
    cmd_thr = thr[thr["mode"] == "command"]
    timing = read_csv(RESULTS / "exp_c" / "timing_summary.csv")
    trans = read_csv(RESULTS / "exp_c" / "transmission_check.csv")
    p3 = a[a.condition == "P3"]
    cmd = o[o["mode"] == "command"]
    h1 = read_json(RESULTS / "exp_a" / "h1.json")
    dec = read_json(RESULTS / "exp_b" / "decision.json")
    sealed = read_json(RESULTS / "exp_c" / "sealed_gate" / "summary.json")
    rej = read_csv(RESULTS / "exp_c" / "sealed_gate" / "rejection_rates.csv")
    rej_e = rej[rej.session == "E"]
    worst = rej_e.loc[rej_e.hold_rate.idxmax()]
    lam1 = p3[p3.target == 1.0].iloc[0]
    tb, tc = "results/exp_b/tradeoff.csv", "results/exp_c/observation_strongest.csv"
    d1 = pd.DataFrame([r for r in tradeoff if r["experiment"] == "exp_d"
                       and r["kind"] == "task" and r["d"] == 1])
    td = "results/exp_d/tradeoff.csv"
    return {
        "d1_identity_range": dict(value=[clean(d1.identity_ba.min()), clean(d1.identity_ba.max())],
                                  source=src(td, kind="task", d=1)),
        "d1_task_range": dict(value=[clean(d1.task_ba.min()), clean(d1.task_ba.max())],
                              source=src(td, kind="task", d=1)),
        "raw_identity": dict(value=clean(cell("P0").identity_ba), source=src(tb, condition="P0")),
        "raw_rc_identity": dict(value=clean(cell("P0-RC").identity_ba),
                                source=src(tb, condition="P0-RC")),
        "bottleneck_identity": dict(value=clean(cell("P2").identity_ba),
                                    source=src(tb, condition="P2")),
        "bottleneck_rc_identity": dict(value=clean(cell("P2-RC").identity_ba),
                                       source=src(tb, condition="P2-RC")),
        "raw_task": dict(value=clean(cell("P0").task_ba), source=src(tb, condition="P0")),
        "bottleneck_rc_task": dict(value=clean(cell("P2-RC").task_ba),
                                   source=src(tb, condition="P2-RC")),
        "head_lambda1": dict(value=clean(lam1.adversary_main_ba),
                             source=src("results/exp_b/adversary_comparison.csv",
                                        condition="P3", target=1.0)),
        "independent_lambda1": dict(value=clean(lam1.independent_ba),
                                    source=src("results/exp_b/adversary_comparison.csv",
                                               condition="P3", target=1.0)),
        "independent_max": dict(value=clean(p3.independent_ba.max()),
                                source=src("results/exp_b/adversary_comparison.csv",
                                           condition="P3")),
        "selected_lambda": dict(value=clean(p3[p3.selected].target.iloc[0]),
                                source=src("results/exp_b/adversary_comparison.csv",
                                           selected=True)),
        "h2_accepted": dict(value=bool(dec["h2_accepted"]), source=src("results/exp_b/decision.json")),
        "cmd_k1_range": dict(value=[clean(cmd[cmd.k == 1].ba_mean.min()),
                                    clean(cmd[cmd.k == 1].ba_mean.max())],
                             source=src(tc, mode="command", k=1)),
        "cmd_k20_range": dict(value=[clean(cmd[cmd.k == 20].ba_mean.min()),
                                     clean(cmd[cmd.k == 20].ba_mean.max())],
                              source=src(tc, mode="command", k=20)),
        "cmd_within5": dict(value=[int((cmd_thr.smallest_k <= 5).sum()), int(len(cmd_thr))],
                            source=src("results/exp_c/observation_threshold.csv",
                                       mode="command", rule="smallest_k <= 5")),
        "chance": dict(value=clean(chance), source=src("configs/exp_c.yaml", key="observation.chance")),
        "h1": dict(value=dict(b3=clean(np.mean([v["B3"] for v in h1["per_subject"].values()])),
                              b2=clean(np.mean([v["B2"] for v in h1["per_subject"].values()])),
                              diff=clean(h1["mean_difference"]), positive=h1["n_positive"],
                              n=h1["n_subjects"], p=clean(h1["p_value"])),
                   source=src("results/exp_a/h1.json")),
        "mae_tau05": dict(value=[clean(ctrl[ctrl.tau == 0.5].mae_deg_mean.min()),
                                 clean(ctrl[ctrl.tau == 0.5].mae_deg_mean.max())],
                          source=src("results/exp_c/control_summary.csv", tau=0.5)),
        "hold_tau08": dict(value=[clean(ctrl[ctrl.tau == 0.8].hold_rate_mean.min()),
                                  clean(ctrl[ctrl.tau == 0.8].hold_rate_mean.max())],
                           source=src("results/exp_c/control_summary.csv", tau=0.8)),
        "transmission": dict(value=dict(passed=int(trans.passed.sum()), settings=int(len(trans)),
                                        mismatches=int(trans.command_mismatches.sum())),
                             source=src("results/exp_c/transmission_check.csv")),
        "timing_median_ms": dict(value=[clean(timing.total_median_ms.min()),
                                        clean(timing.total_median_ms.max())],
                                 source=src("results/exp_c/timing_summary.csv")),
        "sealed_gate_e_hold": dict(value=clean(sealed["e_hold_rate_gate"]),
                                   source=src("results/exp_c/sealed_gate/summary.json")),
        "sealed_gate_t_hold": dict(value=clean(sealed["t_hold_rate_gate"]),
                                   source=src("results/exp_c/sealed_gate/summary.json")),
        "sealed_gate_worst": dict(value=dict(subject=int(worst.subject),
                                             hold=clean(worst.hold_rate)),
                                  source=src("results/exp_c/sealed_gate/rejection_rates.csv",
                                             session="E")),
        "sealed_paths_clean": dict(value=bool(h1["sealed_paths_clean"]),
                                   source=src("results/exp_a/h1.json")),
    }


def scenario_note(rows: list[dict]) -> str:
    """Caption for the observation rows: which strongest values come from the cross-day scenario."""
    cross = [r for r in rows if r["experiment"] == "exp_c" and r["mode"] == "command"
             and r["scenario"] != "main"]
    cells = "; ".join(f"{r['condition']}, τ = {r['tau']:g}, k = {r['k']}: {r['ba_mean']:.3f}"
                      for r in cross)
    return ("Each value is the strongest attacker over attackers and both scenarios: main "
            "(within session E, runs 1-3 vs 4-6) and auxiliary (cross-day, session T to E); the "
            "`scenario` field names the scenario of each value. Command cells whose maximum is "
            f"cross-day: {cells or 'none'}.")


def seal():
    log = track(ROOT / "docs" / "decision_log.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*E seal\*\*: commit `([0-9a-f]{40})`", log)
    if not match:
        raise ValueError("Seal commit not found in docs/decision_log.md")
    return dict(commit=match.group(1), source=src("docs/decision_log.md"))


# ---------------------------------------------------------------- replay (exp C)

def labels_by_trial():
    """Left/right labels of E trials, taken from B's saved metadata (no signal arrays)."""
    path = track(ARTIFACTS / "exp_b" / "final" / "P2" / "seed0_outputs.npz")
    with np.load(path, allow_pickle=False) as z:
        meta = pd.DataFrame({k: z[k] for k in ("subject", "session", "run", "trial", "label")})
    return meta[meta.session == "E"].set_index(["subject", "run", "trial"]).label


def simulate(commands, controller):
    angle, out = 0.0, []
    for c in commands:
        proposed = angle + controller["step_deg"] * c
        if not (c != 0 and abs(proposed) > controller["range_deg"]):
            angle = proposed
        out.append(angle)
    return np.array(out)


def replay(strongest: pd.DataFrame, controller):
    trials = pd.read_csv(track(ARTIFACTS / "exp_c" / "final" / "replay_trials.csv.gz"),
                         usecols=["subject", "run", "trial", "tau", "mode", "command", "angle",
                                  "condition", "seed"])
    preds = pd.read_csv(track(ARTIFACTS / "exp_c" / "final" / "attack_predictions.csv.gz"))
    labels = labels_by_trial()
    conditions = sorted(trials.condition.unique())
    seeds = sorted(trials.seed.unique())
    taus = sorted(trials.tau.unique())
    streams = sorted(preds.stream.unique())

    # Feature and command transmission must give identical commands (results/exp_c/
    # transmission_check.csv); export the command stream once and verify.
    key = ["condition", "seed", "tau", "subject", "run", "trial"]
    cm = trials[trials["mode"] == "command"].set_index(key).sort_index()
    fm = trials[trials["mode"] == "feature"].set_index(key).sort_index()
    if not (cm.command.to_numpy() == fm.command.to_numpy()).all():
        raise ValueError("Feature and command transmission disagree")
    cmd = trials[trials["mode"] == "command"]

    attacker = {(r.condition, r.stream, int(r.k)): (r.attacker, r.scenario)
                for r in strongest.itertuples()}
    preds = preds.set_index(["condition", "stream", "k", "attacker", "scenario"]).sort_index()
    out = {}
    for subject in sorted(cmd.subject.unique()):
        s_cmd = cmd[cmd.subject == subject]
        runs = sorted(s_cmd.run.unique())
        first = s_cmd[(s_cmd.condition == conditions[0]) & (s_cmd.seed == seeds[0])
                      & (s_cmd.tau == taus[0])]
        order = {r: first[first.run == r].trial.to_numpy() for r in runs}  # recorded order
        if any(len(order[r]) < EPISODE for r in runs):
            raise ValueError(f"Subject {subject}: run shorter than one episode")
        lab = {r: "".join("R" if labels.loc[(subject, r, t)] == 1 else "L"
                          for t in order[r][:EPISODE]) for r in runs}
        commands, attacks = {}, {}
        for (c, seed, tau), g in s_cmd.groupby(["condition", "seed", "tau"], sort=True):
            per_run = []
            for r in runs:
                rows = g[g.run == r]
                if not (rows.trial.to_numpy() == order[r]).all():
                    raise ValueError("Replay trial order differs between settings")
                seq = rows.command.to_numpy()[:EPISODE]
                if not np.allclose(simulate(seq, controller), rows.angle.to_numpy()[:EPISODE]):
                    raise ValueError("Arm simulation does not match stored angles")
                per_run.append("".join(CMD[int(v)] for v in seq))
            commands.setdefault(c, {}).setdefault(str(seed), {})[f"{tau:g}"] = per_run
        for c in conditions:
            for stream in streams:
                for k in KS:
                    name, scenario = attacker[(c, stream, k)]
                    block = preds.loc[(c, stream, k, name, scenario)]
                    block = block[block.subject == subject]
                    for seed in seeds:
                        b = block[block.seed == seed]
                        per_run = []
                        for r in runs:
                            if scenario == "main":
                                direction = 0 if r in E_HALVES[0][1] else 1
                                rows = b[(b.run == r) & (b.direction == direction)]
                            else:
                                rows = b[b.run == r]
                            n = len(order[r]) // k
                            if len(rows) != n:
                                raise ValueError(f"Expected {n} groups, got {len(rows)}")
                            # Test groups are non-overlapping and in recorded order
                            # (beat/exp_c.py observation_groups); keep those inside the episode.
                            per_run.append("".join(str(int(p)) for p in
                                                   rows.prediction.to_numpy()[:EPISODE // k]))
                        attacks.setdefault(c, {}).setdefault(str(seed), {}) \
                            .setdefault(stream, {})[str(k)] = per_run
        out[int(subject)] = dict(subject=int(subject), runs=[int(r) for r in runs],
                                 labels=[lab[r] for r in runs], commands=commands,
                                 attacks=attacks)
    meta = dict(conditions=conditions, seeds=[int(s) for s in seeds],
                taus=[float(t) for t in taus], streams=streams, k=list(KS),
                episode_length=EPISODE, controller=controller,
                attacker={f"{c}|{s}|{k}": dict(attacker=a, scenario=sc)
                          for (c, s, k), (a, sc) in attacker.items()},
                note="Episode = first 20 trials of each E run; arm and target restart at 0 deg. "
                     "Attacker guesses come from non-overlapping groups of k consecutive "
                     "outputs inside the episode; attacker and scenario are the strongest "
                     "per cell in results/exp_c/observation_strongest.csv.")
    return out, meta


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay").mkdir(exist_ok=True)
    data = generic()
    status = experiments()
    config_c = yaml.safe_load(track(ROOT / "configs" / "exp_c.yaml").read_text(encoding="utf-8"))
    controller = {k: config_c["controller"][k] for k in ("step_deg", "range_deg", "episode_length",
                                                         "reach_tolerance_deg")}
    control = read_csv(RESULTS / "exp_c" / "control_summary.csv")
    control = control[[c for c in control.columns
                       if not c.endswith(("_median", "_std")) or c.startswith("mae")]]
    strongest = read_csv(RESULTS / "exp_c" / "observation_strongest.csv")
    subjects, replay_meta = replay(strongest, controller)
    # JSON for inspection; the .js copies let index.html run from file:// where fetch is blocked.
    for s, payload in subjects.items():
        text = json.dumps(payload, separators=(",", ":"))
        (OUT / "replay" / f"s{s}.json").write_text(text, encoding="utf-8")
        (OUT / "replay" / f"s{s}.js").write_text(
            f"(window.BEAT_REPLAY = window.BEAT_REPLAY || {{}})[{s}] = {text};\n", encoding="utf-8",
            newline="\n")  # LF on every platform, as in Git (.gitattributes)
    chance = config_c["observation"]["chance"]
    g_data, g_headline = pitfall2()
    seals = timeline(status["exp_l"]["status"])
    results = dict(
        schema=2,
        experiments=status,
        headline={**headline(data["tradeoff"], data["adversary"], data["observation"], chance),
                  **g_headline},
        seal=seal(),
        timeline=seals,
        eegnet=eegnet(chance),
        pitfall2=g_data,
        checklist=checklist(),
        lee2019=lee2019(seals["events"][4]["commit"]),
        k_frontier=k_frontier(),
        tradeoff=dict(rows=data["tradeoff"], source="results/<exp>/tradeoff.csv"),
        adversary=dict(rows=data["adversary"], source="results/<exp>/adversary_comparison.csv"),
        observation=dict(rows=data["observation"],
                         source="results/<exp>/observation_strongest.csv",
                         note=scenario_note(data["observation"])),
        control=dict(rows=records(control, experiment="exp_c"),
                     source="results/exp_c/control_summary.csv"),
        replay=dict(meta=replay_meta, files={str(s): f"data/replay/s{s}.js" for s in subjects},
                    source=["artifacts/exp_c/final/replay_trials.csv.gz",
                            "artifacts/exp_c/final/attack_predictions.csv.gz",
                            "artifacts/exp_b/final/P2/seed0_outputs.npz (labels only)"]),
        sources=dict(sorted(sources.items())),
    )
    text = json.dumps(results, separators=(",", ":"))
    (OUT / "results.json").write_text(text, encoding="utf-8")
    (OUT / "results.js").write_text(f"window.BEAT_DATA = {text};\n", encoding="utf-8",
                                    newline="\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*.js*"))
    print(f"Wrote {OUT} ({total / 1024:.0f} KiB); experiments: "
          + ", ".join(f"{k}={v['status']}/{v['evidence']}" for k, v in status.items()))


if __name__ == "__main__":
    main()
