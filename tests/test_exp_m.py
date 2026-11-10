"""Experiment M: synthetic end-to-end run (dev -> streams -> final) and the seal-4 guards."""
import hashlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from beat.config import ROOT
from beat.data import LeakageError, Trials
from beat.exp_k import fit_eq, rr_matrix, task_information, wrong_weight
from beat.exp_m import (SEAL_FILES, check_data, day1_only, dev_features, import_closure,
                        impute, input_files, load_cfg, main, own_groups, rr_matched_q, run_dev,
                        run_final, run_streams, verify_seal, versions, write_selection)
from beat.yang2025 import member_path

USERS = 6


def synthetic_day(day, rng, mixing, samples=120):
    """Per-user spatial mixing (identity), class-dependent source power (task), day drift."""
    X, rows = [], []
    for u in range(USERS):
        drift = np.eye(19) + 0.05 * day * rng.normal(size=(19, 19))
        labels = rng.permutation(np.repeat([0, 1], 100))
        for t, label in enumerate(labels):
            power = np.ones(19)
            power[[0, 1]] = (3.0, 1.0) if label == 0 else (1.0, 3.0)
            source = rng.normal(size=(19, samples)) * np.sqrt(power)[:, None]
            X.append((drift @ mixing[u] @ source).astype(np.float32))
            rows.append(dict(subject=u + 1, session="T" if day == 1 else "E", run=t // 40 + 1,
                             trial=t + 1, label=int(label), artifact=False,
                             finite=not (u == 0 and t == 7), day=day))
    return Trials(np.stack(X), pd.DataFrame(rows))


@pytest.fixture(scope="module")
def synthetic_cfg():
    cfg = load_cfg()
    cfg.update(subjects=list(range(1, USERS + 1)), seeds=[0], workers=1,
               postprocessing_seeds=[0], tau_star=None, m2=dict(cfg["m2"], status="pending"))
    cfg["training"] = dict(cfg["training"], max_epochs=2)
    cfg["m3"] = dict(cfg["m3"], permutations=200, k_curve=[1, 50], status="pending")
    cfg["power"] = dict(cfg["power"], n_sim=200)
    return cfg


def test_synthetic_end_to_end(tmp_path, synthetic_cfg):
    rng = np.random.default_rng(0)
    mixing = np.eye(19) + 0.4 * rng.normal(size=(USERS, 19, 19))
    days = {d: synthetic_day(d, rng, mixing) for d in (1, 2, 3)}
    out, art = tmp_path / "results", tmp_path / "artifacts"
    cfg = dict(synthetic_cfg, _result_dir=str(out), _artifact_dir=str(art))
    dev = run_dev(days[1], cfg, out, art)
    assert set(dev["budgets"]) == {"common_d8", "rc_d8"}
    assert dev["p3_runtime"]["P3_lambda1"]["full_day1_fit_seconds"] > 0
    assert dev["n_retained"] == USERS * 200 - 1
    dev_features(days[1], cfg, art)
    streams = run_streams(days[1], cfg, out, art)
    selection = write_selection(cfg, out, dev, streams)
    gates = selection["gates"]
    assert 0.5 <= gates["tau"]["tau_star"] <= 0.9
    assert gates["M2"]["status"] in ("confirmatory", "descriptive")
    assert gates["M3"]["status"] in ("confirmatory", "estimate")
    assert 0 <= gates["M3"]["null"]["p"] <= 1 and len(selection["m3_power"]) == 3
    assert all(0 <= v["q"] <= 1 for v in selection["rr_matched"].values())
    # User 6 has no day-3 recording: a missing user-day, never an abort.
    day3 = days[3].subset(days[3].subject != USERS)
    evaluation = Trials(np.concatenate([days[2].X, day3.X]),
                        pd.concat([days[2].meta, day3.meta], ignore_index=True))
    with pytest.raises(LeakageError):  # tau* and claim statuses not yet fixed in the config
        run_final(days[1], evaluation, cfg, selection, out / "x", art)
    frozen = dict(cfg, tau_star=gates["tau"]["tau_star"],
                  m2=dict(cfg["m2"], status=gates["M2"]["status"]),
                  m3=dict(cfg["m3"], status=gates["M3"]["status"]))
    selection = write_selection(frozen, out, dev, streams)
    decision = run_final(days[1], evaluation, frozen, selection, out / "final", art)
    assert decision["holm_family"][0] == "M1" and set(decision["p"]) == set(decision["holm_family"])
    assert decision["missing_user_days"] == [{"subject": USERS, "day": 3}]
    assert decision["M1"]["missing"] == 1 and decision["M3"]["missing"] == 0
    assert all(0 <= p <= 1 for p in decision["p_holm"].values())
    assert decision["M1"]["strongest"] > 1 / USERS  # spatial mixing identifies users
    assert 0 <= decision["M3"]["permutation_p"] <= 1
    est = pd.read_csv(out / "final" / "final_estimates_identification.csv")
    assert set(est.defence) == {"eq", "recal", "rr_q0.5", "rr_matched"}
    assert set(est.scenario) == {"cal_to_dep", "dep_to_dep", "same_day_halves"}
    util = pd.read_csv(out / "final" / "final_estimates_utility.csv")
    cost = util[util.statistic == "utility_privacy_cost"]
    assert set(cost.defence) == {"eq", "rr_q0.5", "rr_matched"} and cost["mean"].notna().all()
    with pytest.raises(LeakageError):
        run_final(days[1], evaluation, dict(frozen, alpha=0.1), selection, out / "y", art)


def test_rr_matched_q_gives_rr_after_recalibration_eqs_utility():
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 30, size=(6, 2, 3)).astype(float) + 1
    users = np.arange(1, 7)
    prior = [0.5, 0.5]
    for tau in (0.5, 0.6):
        w = wrong_weight(tau)
        recal = fit_eq(counts, own_groups(users), "confusion", prior, 0.5, w)[0]
        eq = fit_eq(counts, np.repeat([0, 1], 3), "confusion", prior, 0.5, w)[0]
        u_recal = task_information(counts, recal, w)["utility"]
        u_eq = task_information(counts, eq, w)["utility"]
        q = rr_matched_q(u_recal, u_eq, tau)
        rr = task_information(counts, recal @ rr_matrix(q), w)["utility"]
        assert 0 < q < 1 and np.isclose(rr.mean(), u_eq.mean())


def test_missing_cells_imputed_up_to_two_users():
    v = np.array([0.9, np.nan, 0.8, np.nan])
    (filled,), missing = impute([v], [0.5], 2)
    assert missing == 2 and np.allclose(filled, [0.9, 0.5, 0.8, 0.5])
    assert impute([np.array([np.nan] * 3 + [1.0])], [0.5], 2) == (None, 3)


def test_final_refuses_without_explicit_dirs_or_seal(tmp_path):
    with pytest.raises(LeakageError):
        main(["final", "--seal-commit", "HEAD"])
    with pytest.raises(LeakageError):
        main(["final", "--seal-commit", "HEAD", "--result-dir", str(tmp_path),
              "--artifact-dir", str(tmp_path)])
    with pytest.raises(LeakageError):  # no annotated tag seal-4 in this repository yet
        main(["final", "--seal-commit", "HEAD", "--result-dir", str(ROOT / "results/exp_m"),
              "--artifact-dir", str(ROOT / "artifacts/exp_m")])


def test_day1_guard_blocks_later_days(tmp_path):
    (tmp_path / "day2_sub001.npz").write_bytes(b"x")
    with day1_only() as audit, pytest.raises(LeakageError):
        open(tmp_path / "day2_sub001.npz", "rb")
    assert audit["blocked"]


def test_import_closure_covers_shared_modules():
    closure = import_closure()
    for path in ("beat/__init__.py", "beat/config.py", "beat/data.py", "beat/exp_k.py",
                 "beat/common.py", "beat/yang2025.py", "beat/exp_l.py", "beat/models.py"):
        assert path in closure


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL).decode()


def _commit_seal(repo, message):
    """Commit everything, put annotated tag seal-4 on it and name it in the seal manifest."""
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-f", "-a", "seal-4", "-m", "s")
    commit = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "seals").mkdir(exist_ok=True)
    (repo / "seals/manifest.yaml").write_text(yaml.safe_dump(dict(
        scope=["beat/*.py", "configs/*", "scripts/fetch_yang2025.py"],
        seals=[dict(name="seal-4", tag="seal-4", commit=commit,
                    selections=["results/exp_m/selection.json", "docs/preregistration_m.md",
                                "checksums/yang2025_day1.sha256"])])), encoding="utf-8")
    return commit


def test_verify_seal_checks_the_selection_written_by_write_selection(tmp_path):
    repo = tmp_path / "repo"
    cfg = dict(load_cfg(), thresholds_status="frozen")
    for path in dict.fromkeys(SEAL_FILES + tuple(import_closure())):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == "configs/exp_m.yaml":
            target.write_bytes(yaml.safe_dump(cfg).encode())
        elif path != "results/exp_m/selection.json":
            shutil.copyfile(ROOT / path, target)
    shutil.copyfile(ROOT / ".gitattributes", repo / ".gitattributes")
    streams = dict(gates={}, rr_matched={}, power=pd.DataFrame(), tau_grid=pd.DataFrame(),
                   streams=pd.DataFrame(), summary=pd.DataFrame())
    selection = write_selection(cfg, repo / "results/exp_m", dict(versions=versions()), streams,
                                root=repo)
    assert set(selection["input_sha256"]) == set(input_files(repo))
    assert {"beat/exp_m.py", "beat/exp_k.py", "beat/common.py", "beat/yang2025.py",
            "configs/exp_m.yaml", "configs/exp_k.yaml"} <= set(selection["input_sha256"])
    _git(repo, "init", "-q")
    commit = _commit_seal(repo, "seal")
    assert verify_seal(commit, cfg, root=repo) == commit
    _git(repo, "tag", "-f", "seal-4")  # a lightweight tag is not enough
    with pytest.raises(LeakageError):
        verify_seal(commit, cfg, root=repo)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-f", "-a", "seal-4", "-m", "s")
    with pytest.raises(LeakageError):
        verify_seal(commit, dict(cfg, seeds=[0]), root=repo)
    manifest = (repo / "seals/manifest.yaml").read_text(encoding="utf-8")
    (repo / "seals/manifest.yaml").write_text(manifest.replace(commit, "0" * 40),
                                              encoding="utf-8")
    with pytest.raises(LeakageError, match="manifest"):
        verify_seal(commit, cfg, root=repo)
    (repo / "seals/manifest.yaml").write_text(manifest, encoding="utf-8")
    (repo / "beat/common.py").write_bytes(b"changed\n")
    with pytest.raises(LeakageError):
        verify_seal(commit, cfg, root=repo)
    # K's code changes after the selection was written: sealing the new code with the old
    # selection must fail.
    (repo / "beat/common.py").write_bytes((ROOT / "beat/common.py").read_bytes())
    with open(repo / "beat/exp_k.py", "a", encoding="utf-8", newline="\n") as f:
        f.write("# frozen after review\n")
    later = _commit_seal(repo, "freeze K")
    with pytest.raises(LeakageError, match="selection"):
        verify_seal(later, cfg, root=repo)


def test_check_data_verifies_recorded_hashes_and_lists_missing_user_days(tmp_path):
    cfg = dict(load_cfg(), subjects=[1, 2])
    data = tmp_path / cfg["data_dir"]
    data.mkdir(parents=True)
    (tmp_path / "checksums").mkdir()
    lines = {1: [], 2: []}
    for day, subject in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1)):
        path = data / f"day{day}_sub{subject:03d}.npz"
        path.write_bytes(f"{day}{subject}".encode())
        name = f"{cfg['data_dir']}/{path.name}"
        lines[min(day, 2)] += [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}"] + [
            f"{'0' * 64}  {member_path(subject, day, kind)}" for kind in ("data", "evt")]
    for day, name in ((1, "yang2025_day1.sha256"), (2, "yang2025_day23.sha256")):
        (tmp_path / "checksums" / name).write_text("\n".join(lines[day]) + "\n")
    assert check_data(cfg, root=tmp_path) == [{"subject": 2, "day": 3}]
    (data / "day2_sub002.npz").write_bytes(b"changed")
    with pytest.raises(LeakageError):
        check_data(cfg, root=tmp_path)
