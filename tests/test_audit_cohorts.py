"""scripts/audit_cohorts.py guards on synthetic stand-ins (docs/plan_v3.md §4): output
confinement, the Yang2025 freeze gate, the batch lock and repeat rule, closure after a worker
failure, the pilot's refusal, reconciliation of reports with jobs.json and the bundle, the
bundled-source check, the runtime gate at freeze, the sampling seeds and the Holm decision."""

import copy
from dataclasses import asdict
import importlib.util
import json
import operator
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DESTINATIONS = dict(
    inputs_dir="artifacts/audit/inputs", manifest="results/audit/export_manifest.json",
    utility="results/audit/utility.csv", checks="results/audit/export_checks.json",
    results_dir="results/audit", summary="results/audit/summary.csv",
    attempts="results/audit/attempts.jsonl", parity="results/audit/parity.json",
    runtime="results/audit/runtime_projection.json", v1_selection="results/audit/v1.json",
    jobs_draft="results/audit/jobs_draft.json", freeze_dir="results/audit/freeze",
    batch_lock="artifacts/audit/frozen_batch.lock")
STREAM = "features_P0"
P_VALUES = dict(cosine=0.01, lda_cosine=0.03, pair_logistic=0.2)
AUC = {"same-day": 0.8, "cross-day": 0.7}  # the stand-in report's AUC per condition
EXPORT = "artifacts/audit/inputs/yang2025/features_P0_seed0.npz"


@pytest.fixture(scope="module")
def cohorts():
    spec = importlib.util.spec_from_file_location("audit_cohorts",
                                                  ROOT / "scripts" / "audit_cohorts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def standin(cohorts, tmp_path, monkeypatch):
    """A temporary project root with one synthetic Yang2025 export and its hashed sources."""
    monkeypatch.setattr(cohorts, "ROOT", tmp_path)
    monkeypatch.setattr(cohorts, "pip_freeze", lambda: "numpy==2.0\n")

    def put(rel, data):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return rel

    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(1, 7), 8)
    day, segment = np.tile(np.repeat([1, 2], 4), 6), np.tile([1, 1, 2, 2], 12)
    (tmp_path / EXPORT).parent.mkdir(parents=True)
    np.savez(tmp_path / EXPORT, outputs=rng.normal(size=(48, 3)), user=users, day=day,
             segment=segment, order=np.tile(np.arange(4), 12), target=np.tile([0, 1], 24))
    keys = put(EXPORT.replace(".npz", ".trials.csv"), b"row,user\n0,1\n")
    archive = put("data/processed/yang2025/day1_sub001.npz", b"archive")
    model = put("artifacts/exp_m/final/model_P0_0.joblib", b"model")
    oof = put("artifacts/exp_m/oof_P2-RC_d8_0.joblib", b"cross-fitted")
    for rel in ("beat_audit/__init__.py", "scripts/audit_cohorts.py", "docs/plan_v3.md"):
        put(rel, b"stand-in\n")
    sha = cohorts.sha256_file
    manifest = dict(
        exports=[dict(file=EXPORT, sha256=sha(tmp_path / EXPORT), cohort="yang2025", rung="P0",
                      kind="features", tau=None, model_seed=0, users=list(range(1, 7)),
                      days={"1": "day 1", "2": "day 2"}, segments={"1": "a", "2": "b"},
                      day1_status="in-sample reference", seconds_per_output={"value": 8.0},
                      shape=[48, 3], trial_keys=dict(file=keys, sha256=sha(tmp_path / keys)))],
        yang2025_archives=[dict(path=archive, sha256=sha(tmp_path / archive))],
        sources={model: dict(sha256=sha(tmp_path / model), roles=["M final model"]),
                 oof: dict(sha256=sha(tmp_path / oof), roles=["M day-1 cross-fitted logits"])})
    config = dict(DESTINATIONS, audit=dict(
        settings=dict(k_grid=[1, 2], fold_seed=1, seed=2, permutations=10000, bootstraps=2000,
                      threads=1, comparators=["cosine", "lda_cosine", "pair_logistic"],
                      gallery=True, sequential=True, closed_set=True),
        reductions=[], max_parallel_jobs=1, day_order=dict(yang2025=[1, 2, 3]),
        limits=dict(stream_minutes=60, stream_gb=8, yang2025_batch_hours=3),
        conditions=dict(yang2025={
            "same-day": dict(relation="same-day", pairs=[dict(a=["1/1"], b=["1/2"]),
                                                         dict(a=["2/1"], b=["2/2"])]),
            "cross-day": dict(relation="cross-day", pairs=[dict(a=["1"], b=["2"])])}),
        rung_conditions={}, day1_notes={}),
        # V2 stands in on the same stream: the decision logic does not depend on the kind.
        predictions=dict(alpha=0.05, V1=dict(rung=STREAM, condition="cross-day", k=2,
                                             comparator="cosine", selection=dict(
                                                 cohort="lee2019", rung=STREAM, k=2,
                                                 condition="cross-day",
                                                 order=["cosine", "lda_cosine", "pair_logistic"],
                                                 rule="stand-in")),
                         V2=dict(rung=STREAM, condition="same-day", k=2,
                                 comparator="lda_cosine"),
                         estimate=dict(rung=STREAM, same_day="same-day", cross_day="cross-day")),
        freeze_expect=dict(exports=2, yang2025_archives=1, final_models=1, day1_crossfitted=1))
    put("configs/audit.yaml", yaml.safe_dump(config).encode())
    cfg = cohorts.read_config(tmp_path / "configs/audit.yaml")
    cohorts.write_json(cfg["manifest"], manifest)
    return cfg, manifest


@pytest.mark.parametrize("name, value", [
    ("manifest", "results/exp_m/final_task.csv"), ("inputs_dir", "artifacts/exp_m"),
    ("summary", "results/audit/../exp_l/summary.csv"), ("checks", "results/audit/validation.json"),
    ("freeze_dir", "OUTSIDE")])
def test_destinations_outside_the_audit_trees_are_refused(cohorts, tmp_path, monkeypatch,
                                                          name, value):
    monkeypatch.setattr(cohorts, "ROOT", tmp_path / "project")
    value = str(tmp_path / "elsewhere") if value == "OUTSIDE" else value
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(dict(DESTINATIONS, **{name: value})), encoding="utf-8")
    with pytest.raises(SystemExit, match=f"refused {name}"):
        cohorts.main(["summary", "--config", str(path)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audit.yaml"]
    cfg = dict(DESTINATIONS)
    cohorts.confine(cfg)  # resolves every destination in place
    assert all(Path(cfg[n]).is_relative_to(tmp_path / "project") for n in DESTINATIONS)


def freeze(cohorts, cfg, manifest):
    """The stand-in bundle; returns jobs.json, the frozen hashes reconcile reads and the bundle."""
    cohorts.build_bundle(cfg, manifest, dict(comparator="cosine"), dict(required_reductions=[]))
    folder = Path(cfg["freeze_dir"])
    frozen = dict(exports=cohorts.read_json(folder / "hashes.json")["exports"],
                  sources=cohorts.read_json(folder / "sources.json")["files"])
    return cohorts.read_json(folder / "jobs.json"), frozen, cohorts.read_bundle(cfg)


# The tool's canonical resolved pairs of the stand-in conditions, written out by hand.
RESOLVED = {"same-day": [[[["1", "1"]], [["1", "2"]]], [[["2", "1"]], [["2", "2"]]]],
            "cross-day": [[[["1", "1"], ["1", "2"]], [["2", "1"], ["2", "2"]]]]}


def complete_report(cohorts, document, frozen):
    """A report with everything reconcile requires: model seeds, configuration, hashes, every
    expected cell with its resolved pairs and components in every fold, and the contrasts."""
    from beat_audit.io import digest
    job = document["streams"][0]
    folds = len(set(document["folds"]["assignment"].values()))
    configuration = json.loads(json.dumps(dict(
        settings=cohorts.audit_config(document["settings"], job).to_dict(), kind=job["kind"],
        conditions=[asdict(c) for c in cohorts.conditions_of(job["conditions"])],
        day_order=job["day_order"], day_order_source="explicit")))
    source = {Path(n).name: s for n, s in frozen["sources"].items() if n.startswith("beat_audit/")}
    cells = []
    for condition, k in sorted({(c["condition"], c["k"]) for c in document["cells"]}):
        pairs = job["conditions"][condition]["pairs"]
        comparators = {}
        for c in document["cells"]:
            if (c["condition"], c["k"]) != (condition, k):
                continue
            ops = [dict(fold=f, model_seed=x["model_seed"], direction=x["direction"],
                        pair=next(i for i, p in enumerate(pairs)
                                  if (p["a"], p["b"]) == (x["side_a"], x["side_b"])))
                   for f in range(folds) for x in c["components"]]
            comparators[c["comparator"]] = dict(
                auc=AUC[condition], p=P_VALUES[c["comparator"]], interval=[0.6, 0.9],
                draws=10000, operating_points=ops,
                comparison_bounds=dict(components=len(c["components"])))
        cells.append(dict(condition=condition, relation=condition, k=k, status="ok",
                          days=["1", "2"], analysed_users=[str(u) for u in range(1, 7)],
                          excluded_users=[], resolved_pairs=RESOLVED[condition],
                          comparators=comparators, population_hash="h"))
    return dict(
        model_seeds=["0"], configuration=configuration, cells=cells, smallest_detected_k={},
        hashes=dict(source=dict(sha256=digest(source), files=source),
                    configuration=digest(configuration),
                    inputs={"0": frozen["exports"][EXPORT]}),
        contrasts=[dict(same_day="same-day", cross_day="cross-day", k=k, status="ok",
                        comparators={n: dict(mean=AUC["same-day"] - AUC["cross-day"],
                                             interval=[0.01, 0.19]) for n in P_VALUES})
                   for k in (1, 2)])


ATTEMPT = dict(attempt="a", kind="frozen_batch")


def test_yang2025_audits_are_refused_without_a_matching_freeze(cohorts, standin):
    cfg, manifest = standin
    job = cohorts.audit_jobs(cfg, ["yang2025"], manifest)[0]
    with pytest.raises(SystemExit, match="only inside the frozen batch"):
        cohorts.audit_job(cfg, job)
    with pytest.raises(SystemExit, match="only inside the frozen batch"):
        cohorts.develop(cfg, SimpleNamespace(cohort=["yang2025"], rung=None))
    forged = dict(kind="frozen_batch", attempt="x", bundle_sha256="0" * 64)
    result = cohorts.audit_job(cfg, job, forged)  # a frozen kind outside a running batch
    assert result["status"] == "failed" and "no running frozen batch" in result["error"]
    with cohorts.batch_lock(cfg):
        assert cohorts.frozen_mismatches(cfg, job, forged) == ["no freeze bundle"]
    with pytest.raises(SystemExit, match="no freeze bundle"):
        cohorts.frozen_batch(cfg, SimpleNamespace())
    _, _, bundle = freeze(cohorts, cfg, manifest)
    assert cohorts.verify_bundle(cfg, bundle) == []
    with cohorts.batch_lock(cfg):
        assert cohorts.frozen_mismatches(cfg, job, forged) == [
            "bundle: the attempt names another bundle"]
        assert cohorts.frozen_mismatches(cfg, job, dict(forged, bundle_sha256=bundle["sha256"])) == []
    for rel in (EXPORT, "data/processed/yang2025/day1_sub001.npz", "beat_audit/__init__.py"):
        with (cohorts.ROOT / rel).open("ab") as handle:
            handle.write(b"changed")
    mismatches = cohorts.verify_bundle(cfg, cohorts.read_bundle(cfg))
    assert {m.split(": ", 1)[1] for m in mismatches} >= {
        EXPORT, "data/processed/yang2025/day1_sub001.npz", "beat_audit/__init__.py"}
    with pytest.raises(SystemExit, match="mismatches against the freeze bundle"):
        cohorts.frozen_batch(cfg, SimpleNamespace())
    assert not Path(cfg["attempts"]).exists()
    assert not (Path(cfg["results_dir"]) / "yang2025").exists()
    assert not (cohorts.ROOT / job["out"]).exists()


def test_holm_family_of_two_counts_a_non_evaluable_prediction_as_p_one(cohorts):
    both = cohorts.holm_decision({"V1": 0.01, "V2": 0.04}, 0.05)
    assert both["V1"]["holm_p"] == pytest.approx(0.02)
    assert both["V2"]["holm_p"] == pytest.approx(0.04)
    assert both["V1"]["supported"] and both["V2"]["supported"]
    step = cohorts.holm_decision({"V1": 0.02, "V2": 0.30}, 0.05)
    assert step["V1"]["supported"] and not step["V2"]["supported"]
    assert step["V1"]["holm_p"] == pytest.approx(0.04)
    monotone = cohorts.holm_decision({"V1": 0.026, "V2": 0.02}, 0.05)
    assert monotone["V1"]["holm_p"] == monotone["V2"]["holm_p"] == pytest.approx(0.04)
    missing = cohorts.holm_decision({"V1": 0.03, "V2": None}, 0.05)
    assert missing["V2"] == dict(p=1.0, evaluable=False, holm_p=1.0, supported=False)
    assert missing["V1"]["holm_p"] == pytest.approx(0.06) and not missing["V1"]["supported"]


def test_complete_reports_decide_and_a_non_evaluable_prediction_counts_as_p_one(cohorts,
                                                                                 standin):
    cfg, manifest = standin
    document, frozen, _ = freeze(cohorts, cfg, manifest)
    report = complete_report(cohorts, document, frozen)
    statuses = cohorts.reconcile(document, {STREAM: report}, frozen)
    assert len(statuses) == 12 and {s["status"] for s in statuses} == {"ok"}
    decision = cohorts.decision_document(cfg, {STREAM: report}, statuses, ATTEMPT)
    v1, v2 = decision["predictions"]["V1"], decision["predictions"]["V2"]
    assert not decision["technical_failure"] and decision["estimate"]["status"] == "ok"
    assert (v1["p"], v1["holm_p"], v1["supported"], v1["wording"]) == (
        0.01, pytest.approx(0.02), True, "linkage detected")
    assert (v2["holm_p"], v2["supported"]) == (pytest.approx(0.03), True)
    # V1's cell not evaluable: the tool pairs no contrast for it, V1 stays in the family at p = 1
    cell = next(c for c in report["cells"] if (c["condition"], c["k"]) == ("cross-day", 2))
    cell.update(status="not_evaluable", reason="k=2, user: 2/6 excluded", comparators={},
                analysed_users=[str(u) for u in range(3, 7)], excluded_users=["1", "2"])
    report["contrasts"] = [c for c in report["contrasts"] if c["k"] != 2]
    statuses = cohorts.reconcile(document, {STREAM: report}, frozen)
    decision = cohorts.decision_document(cfg, {STREAM: report}, statuses, ATTEMPT)
    v1, v2 = decision["predictions"]["V1"], decision["predictions"]["V2"]
    assert not decision["technical_failure"]
    assert (v1["status"], v1["p"], v1["evaluable"], v1["supported"]) == (
        "not_evaluable", 1.0, False, False)
    assert v1["wording"].startswith("not evaluable")
    assert (v2["holm_p"], v2["supported"]) == (pytest.approx(0.06), False)
    assert decision["estimate"]["status"] == "not_evaluable"


def test_incomplete_reports_are_a_technical_failure_without_a_supported_prediction(cohorts,
                                                                                    standin):
    """The reviewer's case: cells with AUC and p, but no component evidence, hashes or contrasts."""
    cfg, manifest = standin
    document, frozen, _ = freeze(cohorts, cfg, manifest)
    bare = dict(cells=[dict(condition=c, relation=c, k=k, status="ok", excluded_users=[],
                            analysed_users=[str(u) for u in range(1, 7)], population_hash="h",
                            comparators={n: dict(auc=0.9, p=0.0001, interval=[0.8, 1.0],
                                                 draws=10000) for n in P_VALUES})
                       for c in ("same-day", "cross-day") for k in (1, 2)])
    statuses = cohorts.reconcile(document, {STREAM: bare}, frozen)
    assert {s["status"] for s in statuses} == {"failed"}
    assert "source hashes differ" in statuses[0]["reason"]
    decision = cohorts.decision_document(cfg, {STREAM: bare}, statuses, ATTEMPT)
    assert decision["technical_failure"] and decision["estimate"]["status"] == "failed"
    for value in decision["predictions"].values():
        assert value["status"] == "failed" and not value["supported"]
        assert value["wording"].startswith("technical failure")


@pytest.mark.parametrize("breach, reason", [
    (lambda r: r.update(model_seeds=["0", "1"]), "model seeds"),
    (lambda r: r["hashes"]["inputs"].update({"0": "0" * 64}), "input hashes"),
    (lambda r: r["hashes"]["source"]["files"].update({"__init__.py": "0" * 64}), "source hashes"),
    (lambda r: r["configuration"]["settings"].update(seed=3), "configuration hash"),
    (lambda r: r["cells"].append(copy.deepcopy(r["cells"][0])), "reported 2 times"),
    (lambda r: r.pop("contrasts"), None)])
def test_each_breach_of_the_execution_contract_is_a_technical_failure(cohorts, standin, breach,
                                                                      reason):
    cfg, manifest = standin
    document, frozen, _ = freeze(cohorts, cfg, manifest)
    report = complete_report(cohorts, document, frozen)
    breach(report)
    statuses = cohorts.reconcile(document, {STREAM: report}, frozen)
    if reason:
        assert {s["status"] for s in statuses} == {"failed"}
        assert reason in statuses[0]["reason"]
    decision = cohorts.decision_document(cfg, {STREAM: report}, statuses, ATTEMPT)
    assert decision["technical_failure"]
    assert not any(v["supported"] for v in decision["predictions"].values())


def test_a_cell_needs_its_day_pairs_and_every_component_in_every_fold(cohorts, standin):
    cfg, manifest = standin
    document, frozen, _ = freeze(cohorts, cfg, manifest)
    cases = [
        ("resolved day pairs differ", lambda c, v: c.update(resolved_pairs=RESOLVED["same-day"])),
        ("components differ", lambda c, v: v.update(operating_points=[  # one direction only
            op for op in v["operating_points"] if op["direction"] == "A->B"])),
        ("components differ", lambda c, v: v.update(operating_points=[  # one fold only
            op for op in v["operating_points"] if op["fold"] == 0])),
        ("component count differs", lambda c, v: v.update(comparison_bounds=dict(components=1))),
        ("population differs", lambda c, v: c.update(excluded_users=["1"])),
        ("comparator missing", lambda c, v: c["comparators"].pop("cosine"))]
    for reason, breach in cases:
        report = complete_report(cohorts, document, frozen)
        cell = next(c for c in report["cells"] if (c["condition"], c["k"]) == ("cross-day", 2))
        breach(cell, cell["comparators"]["cosine"])
        statuses = {(s["condition"], s["k"], s["comparator"]): s for s in cohorts.reconcile(
            document, {STREAM: report}, frozen)}
        assert reason in statuses["cross-day", 2, "cosine"]["reason"]
        assert statuses["same-day", 2, "cosine"]["status"] == "ok"
        decision = cohorts.decision_document(cfg, {STREAM: report}, list(statuses.values()),
                                             ATTEMPT)
        assert decision["technical_failure"]
        assert decision["predictions"]["V1"]["wording"].startswith("technical failure")


def test_one_batch_at_a_time_and_a_repeat_only_after_a_recorded_technical_failure(cohorts,
                                                                                  standin):
    cfg, manifest = standin
    _, _, bundle = freeze(cohorts, cfg, manifest)
    with cohorts.batch_lock(cfg):  # a batch is running
        assert cohorts.batch_running(cfg)
        with pytest.raises(SystemExit, match="is running"):
            cohorts.frozen_batch(cfg, SimpleNamespace())
        with pytest.raises(SystemExit, match="is running"):
            cohorts.recover_batch(cfg, SimpleNamespace(attempt="a1"))
    assert not cohorts.batch_running(cfg)
    start = dict(attempt="a1", kind="frozen_batch", command="frozen-batch", event="start",
                 bundle_sha256=bundle["sha256"])
    cohorts.log_attempt(cfg, start)  # interrupted: no end record
    with pytest.raises(SystemExit, match="a1 has no end record"):
        cohorts.frozen_batch(cfg, SimpleNamespace())
    with pytest.raises(SystemExit, match="not an open frozen attempt"):
        cohorts.recover_batch(cfg, SimpleNamespace(attempt="a2"))
    assert cohorts.recover_batch(cfg, SimpleNamespace(attempt="a1")) == 0
    end = cohorts.read_attempts(cfg)[-1]
    assert (end["attempt"], end["event"], end["status"]) == ("a1", "end", "technical_failure")
    assert end["recovered"]["command"] == "recover-batch"
    decision = cohorts.read_json(Path(cfg["results_dir"]) / "yang2025" / "decision.json")
    assert decision["technical_failure"]
    assert not any(v["supported"] for v in decision["predictions"].values())
    assert cohorts.batch_kind(cohorts.read_attempts(cfg), bundle) == "technical_rerun"
    cohorts.log_attempt(cfg, dict(start, attempt="a3", kind="technical_rerun"))
    cohorts.log_attempt(cfg, dict(attempt="a3", kind="technical_rerun", command="frozen-batch",
                                  event="end", status="completed"))
    with pytest.raises(SystemExit, match="a3 ended 'completed'"):
        cohorts.frozen_batch(cfg, SimpleNamespace())
    other = dict(start, attempt="a4", bundle_sha256="0" * 64)
    assert pytest.raises(SystemExit, cohorts.batch_kind, [other], bundle).match("another bundle")


def test_run_pool_turns_a_worker_exception_into_a_failed_result(cohorts):
    results = cohorts.run_pool(operator.truediv, [(1, 2), (1, 0)], 2,
                               failed=lambda call, exc: (call, type(exc).__name__))
    assert results == [0.5, ((1, 0), "ZeroDivisionError")]


def test_a_worker_failure_still_closes_every_cell_and_the_attempt(cohorts, standin,
                                                                  monkeypatch):
    cfg, manifest = standin
    document, _, _ = freeze(cohorts, cfg, manifest)
    out = Path(cfg["results_dir"]) / "yang2025"

    def broken(function, calls, count, failed=None):
        raise RuntimeError("worker process died")

    monkeypatch.setattr(cohorts, "run_pool", broken)
    with pytest.raises(RuntimeError, match="worker process died"):
        cohorts.frozen_batch(cfg, SimpleNamespace())
    start, end = cohorts.read_attempts(cfg)
    assert (start["event"], end["event"], end["status"]) == ("start", "end", "technical_failure")
    assert "worker process died" in end["problems"][0]
    cells = cohorts.read_json(out / "cells.json")["cells"]
    assert len(cells) == document["cell_count"] and {c["status"] for c in cells} == {"failed"}
    decision = cohorts.read_json(out / "decision.json")
    assert decision["technical_failure"]
    assert decision["predictions"]["V1"]["wording"].startswith("technical failure")
    assert not cohorts.batch_running(cfg)
    monkeypatch.setattr(cohorts, "run_pool", lambda function, calls, count, failed: [
        failed(call, MemoryError("killed")) for call in calls])
    assert cohorts.frozen_batch(cfg, SimpleNamespace()) == 1  # a technical rerun, again failed
    rerun = cohorts.read_attempts(cfg)[-1]
    assert (rerun["kind"], rerun["status"]) == ("technical_rerun", "technical_failure")
    cells = cohorts.read_json(out / "cells.json")["cells"]
    assert {c["reason"] for c in cells} == {"MemoryError: killed"}


def test_the_pilot_refuses_a_yang2025_job(cohorts, standin, monkeypatch):
    import beat_audit
    cfg, manifest = standin
    monkeypatch.setattr(beat_audit, "audit_timing", lambda *a, **k: pytest.fail("pilot ran"))
    job = cohorts.audit_jobs(cfg, ["yang2025"], manifest)[0]
    with pytest.raises(SystemExit, match="only inside the frozen batch"):
        cohorts.pilot_job(cfg, job, cohorts.audit_settings(cfg))
    assert not Path(cfg["attempts"]).exists()


def test_the_frozen_route_imports_no_project_module_outside_the_bundle():
    code = textwrap.dedent("""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("audit_cohorts", sys.argv[1])
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        import beat_audit, beat_audit.engine, beat_audit.io, beat_audit.protocol
        import beat_audit.resources
        cfg = m.read_config(m.ROOT / "configs/audit.yaml")
        print(json.dumps(dict(
            beat=sorted(n for n in sys.modules if n.split(".")[0] == "beat"),
            unbundled=[n for n in m.local_modules() if n not in m.source_files(cfg)])))
    """)
    run = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/audit_cohorts.py")],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(run.stdout.splitlines()[-1]) == dict(beat=[], unbundled=[])


def test_a_helper_module_outside_the_bundle_is_detected(cohorts, standin, monkeypatch):
    cfg, manifest = standin
    _, _, bundle = freeze(cohorts, cfg, manifest)
    helper = cohorts.ROOT / "beat" / "common.py"
    helper.parent.mkdir()
    helper.write_text("def write_json(path, value): pass\n", encoding="utf-8")
    module = ModuleType("helper_standin")
    module.__file__ = str(helper)
    monkeypatch.setitem(sys.modules, "helper_standin", module)
    assert cohorts.verify_bundle(cfg, bundle) == ["unbundled local module: beat/common.py"]
    job = cohorts.audit_jobs(cfg, ["yang2025"], manifest)[0]
    attempt = dict(ATTEMPT, bundle_sha256=bundle["sha256"])
    with cohorts.batch_lock(cfg):
        assert cohorts.frozen_mismatches(cfg, job, attempt) == [
            "unbundled local module: beat/common.py"]
        with pytest.raises(SystemExit, match="is running"):
            cohorts.frozen_batch(cfg, SimpleNamespace())
    monkeypatch.delitem(sys.modules, "helper_standin")
    assert cohorts.verify_bundle(cfg, bundle) == []


def gate_inputs(cohorts, cfg, ratios):
    """A runtime projection and a completed development run whose command streams ran at
    `ratios` x their projected 10 minutes; the Yang2025 stream is projected at 50 minutes."""
    streams = [dict(cohort="lee2019", rung=f"commands_{i}", pilot="commands", minutes=10.0,
                    peak_gb=1.0) for i in range(len(ratios))]
    streams.append(dict(cohort="yang2025", rung=STREAM, pilot="commands", minutes=50.0,
                        peak_gb=1.0))
    cohorts.write_json(cfg["runtime"], dict(
        required_reductions=[], settings_measured=dict(permutations=10000),
        projections=[dict(reductions=[], streams=streams)]))
    cohorts.log_attempt(cfg, dict(attempt="d", kind="development", command="develop",
                                  event="start", reductions=[]))
    cohorts.log_attempt(cfg, dict(attempt="d", kind="development", command="develop",
                                  event="end", status="completed", results=[
                                      dict(cohort="lee2019", rung=f"commands_{i}",
                                           seconds=600 * r, peak_rss_bytes=2**30)
                                      for i, r in enumerate(ratios)]))


def test_freeze_requires_the_adjusted_projection_within_the_limits(cohorts, standin):
    cfg, _ = standin
    args = SimpleNamespace(replace=False, dry_run=True)
    gate_inputs(cohorts, cfg, [4.8, 1.0])  # the largest ratio applies, not the last one read
    check = cohorts.development_check(cfg, cohorts.read_json(cfg["runtime"]))
    assert check["factor_by_type"] == {"commands": 4.8}
    assert check["yang2025_batch_hours"] == pytest.approx(4.0)
    with pytest.raises(SystemExit, match="exceeds the limits"):
        cohorts.freeze(cfg, args)
    gate_inputs(cohorts, cfg, [1.1, 0.5])  # within limits: freeze proceeds to the V1 rule
    with pytest.raises(SystemExit, match="no development report"):
        cohorts.freeze(cfg, args)
    assert not Path(cfg["freeze_dir"]).exists()


def test_an_adjusted_projection_over_the_limits_applies_the_next_reduction(cohorts, standin,
                                                                          monkeypatch):
    """The development run (every comparator) took 1.5 x its projection, which takes the
    Yang2025 stream to 75 minutes; the gate applies the plan's first reduction, projected at 30
    minutes (45 adjusted), and V1's rule then compares only the surviving comparators."""
    cfg, _ = standin
    gate_inputs(cohorts, cfg, [1.5])
    reduced = []

    def projection(cfg_, manifest, pilots, reductions, permutations):
        reduced.append(list(reductions))
        streams = cohorts.read_json(cfg["runtime"])["projections"][0]["streams"]
        return dict(reductions=list(reductions), streams=[
            dict(s, minutes=30.0) if s["cohort"] == "yang2025" else s for s in streams])

    monkeypatch.setattr(cohorts, "gate_projection", projection)
    monkeypatch.setattr(cohorts, "stored_pilots", lambda *a: {})
    check = cohorts.runtime_check(cfg)
    assert reduced == [["drop_pair_logistic"]]
    assert (check["calibrated_on"], check["reductions"]) == ([], ["drop_pair_logistic"])
    assert check["within_limits"] and check["yang2025_batch_hours"] == pytest.approx(0.75)
    document = cohorts.read_json(cfg["runtime"])
    assert document["required_reductions"] == ["drop_pair_logistic"]
    assert document["development_reductions"] == [dict(
        attempt="d", exceeds=[f"yang2025/{STREAM}: 75.0 min"], applied="drop_pair_logistic")]
    args = SimpleNamespace(replace=False, dry_run=True)
    with pytest.raises(SystemExit, match="differ from the runtime gate's required reductions"):
        cohorts.freeze(cfg, args)
    # V1's rule on a development report in which the removed comparator scored highest
    real_jobs, lee = cohorts.audit_jobs, dict(cohort="lee2019", rung=STREAM,
                                              out="results/audit/lee2019/features_P0")
    monkeypatch.setattr(cohorts, "audit_jobs", lambda c, names, manifest=None: (
        [lee] if list(names) == ["lee2019"] else real_jobs(c, names, manifest)))
    aucs = dict(cosine=0.9, lda_cosine=0.8, pair_logistic=0.99)
    cell = dict(condition="cross-day", k=2, status="ok", comparators={
        n: dict(auc=a, interval=[a - 0.1, a]) for n, a in aucs.items()})
    cohorts.write_json(cohorts.ROOT / lee["out"] / "report.json", dict(cells=[cell]))
    cfg["predictions"]["V1"]["selection"]["order"] = list(aucs)
    assert cohorts.select_v1(cfg)["comparator"] == "pair_logistic"
    cfg["audit"]["reductions"] = ["drop_pair_logistic"]
    selection = cohorts.select_v1(cfg)
    assert selection["comparator"] == "cosine" and list(selection["not_compared"]) == [
        "pair_logistic"]
    assert cohorts.freeze(cfg, args) == 0
    del cell["comparators"]["lda_cosine"]
    cohorts.write_json(cohorts.ROOT / lee["out"] / "report.json", dict(cells=[cell]))
    with pytest.raises(SystemExit, match=r"lacks the surviving comparators \['lda_cosine'\]"):
        cohorts.select_v1(cfg)
    assert not Path(cfg["freeze_dir"]).exists()


def test_the_projection_runs_yang2025_streams_in_jobs_json_order(cohorts):
    streams = [dict(cohort="yang2025", minutes=m) for m in (30, 30, 60)]
    assert cohorts.yang_batch_hours(streams, 2) == pytest.approx(1.5)  # longest-first: 1.0


def test_the_bundle_lists_every_sampling_seed(cohorts, standin):
    cfg, manifest = standin
    document, _, bundle = freeze(cohorts, cfg, manifest)
    assert "seeds.json" in bundle["files"]
    assert document == json.loads(json.dumps(cohorts.jobs_document(cfg, draft=False)))
    seeds = cohorts.read_json(Path(cfg["freeze_dir"]) / "seeds.json")
    assert (seeds["matching_permutations"], seeds["bootstrap_and_contrasts"]) == (2, 3)
    stream, = seeds["streams"]
    assert stream["pairs_per_condition"] == {"same-day": 2, "cross-day": 1}
    assert len(stream["components"]) == 2  # two folds x one model seed
    fold = next(c for c in stream["components"] if c["fold"] == 1)
    assert fold["attack_seed"] == 2 + 1009
    assert [d["seed"] for d in fold["draws"]] == [1011, 1011 + 17, 1011 + 37, 1011 + 37 + 17]


def test_the_sampling_seeds_follow_the_engines_model_seed_order(cohorts, tmp_path, monkeypatch):
    """The reviewer's case: inputs listed as "2", "0", "10" with base seed 6723. beat_audit fits
    them in sorted string order ("0", "10", "2"); seeds.json must give the seeds the engine
    actually uses, recorded here from a run on synthetic streams."""
    from beat_audit import AuditConfig, Condition, Population, Segment, audit_stream, engine
    from beat_audit.synthetic import stream_arrays
    inputs = {}
    for label in ("2", "0", "10"):
        inputs[label] = str(tmp_path / f"seed-{label}.npz")
        np.savez(inputs[label], **stream_arrays(6, 2, 4, seed=int(label)))
    users = [f"u{u:03}" for u in range(6)]
    assignment = {u: i % 2 for i, u in enumerate(users)}
    document = dict(settings=dict(seed=6723, fold_seed=1),
                    folds=dict(assignment=assignment, sampling_seeds=dict(derived="")),
                    streams=[dict(rung="r", inputs=inputs, conditions={
                        "same": dict(pairs=[0, 1]), "cross": dict(pairs=[0])})])
    stream, = cohorts.sampling_seeds(document)["streams"]
    listed = {(c["fold"], c["model_seed"]): c["attack_seed"] for c in stream["components"]}
    assert {m: s for (f, m), s in listed.items() if f == 0} == {"0": 6723, "10": 6824, "2": 6925}
    draws = {}
    for c in stream["components"]:
        draws[c["attack_seed"]] = {d["seed"] for d in c["draws"]}

    drawn = {}

    class Recording(engine.Comparator):
        def fit(self, *args, **kwargs):
            super().fit(*args, **kwargs)
            self.info["seed"] = self.seed
            return self

    def tables(comparator, scoring, targets, mcap, ncap, seed, **kwargs):
        drawn.setdefault(comparator.seed, set()).add(seed)
        return real_tables(comparator, scoring, targets, mcap, ncap, seed, **kwargs)

    real_tables = engine._tables
    monkeypatch.setattr(engine, "Comparator", Recording)
    monkeypatch.setattr(engine, "_tables", tables)
    same = tuple(((Segment(d, "1"),), (Segment(d, "2"),)) for d in ("1", "2"))
    report = audit_stream(
        inputs, conditions=(Condition("same", "same-day", same),
                            Condition("cross", "cross-day", (((Segment("1"),), (Segment("2"),)),))),
        config=AuditConfig(k_grid=(1,), seed=6723, permutations=99, bootstraps=99,
                           comparators=("cosine",), gallery=False, sequential=False,
                           closed_set=False, threads=1),
        population=Population(users=users, folds=assignment))
    assert [c["status"] for c in report["cells"]] == ["ok", "ok"]
    used = {(t["fold"], t["model_seed"]): t["seed"] for c in report["cells"]
            for t in c["comparators"]["cosine"]["training"]}
    assert used == listed
    assert drawn == draws


def test_summaries_before_the_freeze_never_read_a_yang2025_report(cohorts, standin, monkeypatch):
    """The reviewer's case: a Yang2025 report on disk when the development summary is written;
    only the frozen batch summarises Yang2025 reports, those whose stream passed reconciliation."""
    cfg, manifest = standin
    job = cohorts.audit_jobs(cfg, ["yang2025"], manifest)[0]
    report_path = cohorts.ROOT / job["out"] / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{not a report", encoding="utf-8")
    opened, read_json = [], cohorts.read_json
    monkeypatch.setattr(cohorts, "read_json", lambda path: opened.append(Path(path)) or
                        read_json(path))
    assert cohorts.write_summary(cfg) == []
    assert cohorts.summary(cfg, SimpleNamespace()) == 0
    cfg["predictions"]["V1"]["selection"]["cohort"] = "yang2025"
    with pytest.raises(SystemExit, match="selected on a development cohort"):
        cohorts.select_v1(cfg)
    assert report_path.resolve() not in {p.resolve() for p in opened}
    development = Path(cfg["summary"]).read_bytes()
    # the frozen batch's own summary: only reconciled streams
    cfg["predictions"]["V1"]["selection"]["cohort"] = "lee2019"
    document, frozen, bundle = freeze(cohorts, cfg, manifest)
    out = Path(cfg["results_dir"]) / "yang2025"
    for breach, rows in ((None, 12), ("source", 0)):
        report = complete_report(cohorts, document, frozen)
        if breach:
            report["hashes"]["source"]["files"]["__init__.py"] = "0" * 64
        cohorts.write_json(report_path, report)
        record = cohorts.start_attempt(cfg, "frozen_batch", "frozen-batch")
        attempt = dict(attempt=record["attempt"], kind="frozen_batch",
                       bundle_sha256=bundle["sha256"])
        cohorts.close_batch(cfg, document, record, attempt,
                            [dict(rung=STREAM, cohort="yang2025", status="completed")], [])
        text = (out / "summary.csv").read_text(encoding="utf-8")
        assert text.count(f"yang2025,{STREAM}") == rows
    assert Path(cfg["summary"]).read_bytes() == development


@pytest.mark.parametrize("change, reason", [
    (dict(interval=[None, None]), "interval"), (dict(interval=[0.19, 0.01]), "interval"),
    (dict(mean=0.91), "mean"), (dict(mean=float("nan")), "mean")])
def test_an_invalid_contrast_is_a_technical_failure(cohorts, standin, change, reason):
    """The reviewer's cases: an interval [None, None], and a mean of 0.91 beside the cells' AUCs
    of 0.8 (same day) and 0.7 (cross day)."""
    cfg, manifest = standin
    document, frozen, _ = freeze(cohorts, cfg, manifest)
    report = complete_report(cohorts, document, frozen)
    next(c for c in report["contrasts"] if c["k"] == 2)["comparators"]["cosine"].update(change)
    statuses = cohorts.reconcile(document, {STREAM: report}, frozen)
    assert {s["status"] for s in statuses} == {"ok"}
    decision = cohorts.decision_document(cfg, {STREAM: report}, statuses, ATTEMPT)
    estimate = decision["estimate"]
    assert estimate["status"] == "failed" and f"contrast {reason}" in estimate["reason"]
    assert decision["technical_failure"]
    for value in decision["predictions"].values():
        assert not value["supported"] and value["wording"].startswith("no verdict")


def test_a_prediction_outside_the_manifest_fails_the_batch_and_the_freeze(cohorts, standin):
    """The reviewer's case: V2 at k = 999, which jobs.json does not list."""
    cfg, manifest = standin
    document, frozen, _ = freeze(cohorts, cfg, manifest)
    report = complete_report(cohorts, document, frozen)
    statuses = cohorts.reconcile(document, {STREAM: report}, frozen)
    cfg["predictions"]["V2"]["k"] = 999
    decision = cohorts.decision_document(cfg, {STREAM: report}, statuses, ATTEMPT)
    v1, v2 = decision["predictions"]["V1"], decision["predictions"]["V2"]
    assert decision["technical_failure"] and decision["estimate"]["status"] == "ok"
    assert v2["status"] == "failed" and v2["wording"].startswith("technical failure")
    assert not v1["supported"] and v1["wording"].startswith("no verdict")
    args = SimpleNamespace(replace=True, dry_run=True)
    with pytest.raises(SystemExit, match=r"not listed exactly once .*V2 .*999"):
        cohorts.freeze(cfg, args)
    cfg["predictions"]["V2"]["k"] = 2
    cfg["predictions"]["estimate"]["same_day"] = "same-day-d2-d3"
    with pytest.raises(SystemExit, match=r"not listed exactly once .*estimate same_day"):
        cohorts.freeze(cfg, args)


def test_v1_is_selected_on_lee2019_over_every_surviving_comparator(cohorts, standin, monkeypatch):
    """The reviewer's cases: a selection moved to the 2a report, and a selection order that
    leaves out a surviving comparator; both must be refused, never silently applied."""
    cfg, _ = standin
    real_jobs = cohorts.audit_jobs
    reports = {name: dict(cohort=name, rung=STREAM, out=f"results/audit/{name}/features_P0")
               for name in ("lee2019", "bnci2014_001")}
    monkeypatch.setattr(cohorts, "audit_jobs", lambda c, names, manifest=None: (
        [reports[list(names)[0]]] if list(names)[0] in reports
        else real_jobs(c, names, manifest)))
    for name, aucs in (("lee2019", dict(cosine=0.91, lda_cosine=0.99, pair_logistic=0.5)),
                       ("bnci2014_001", dict(cosine=0.99, lda_cosine=0.5, pair_logistic=0.5))):
        cell = dict(condition="cross-day", k=2, status="ok", comparators={
            n: dict(auc=a, interval=[a - 0.1, a]) for n, a in aucs.items()})
        cohorts.write_json(cohorts.ROOT / reports[name]["out"] / "report.json",
                           dict(cells=[cell]))
    selection = cfg["predictions"]["V1"]["selection"]
    assert cohorts.select_v1(cfg)["comparator"] == "lda_cosine"
    selection["cohort"] = "bnci2014_001"
    with pytest.raises(SystemExit, match="V1's own cell on lee2019"):
        cohorts.select_v1(cfg)
    selection.update(cohort="lee2019", k=1)
    with pytest.raises(SystemExit, match="V1's own cell on lee2019"):
        cohorts.select_v1(cfg)
    selection.update(k=2, order=["cosine"])
    with pytest.raises(SystemExit, match="every configured feature comparator once"):
        cohorts.select_v1(cfg)
    selection["order"] = ["lda_cosine", "cosine", "pair_logistic"]
    with pytest.raises(SystemExit, match="every configured feature comparator once"):
        cohorts.select_v1(cfg)
