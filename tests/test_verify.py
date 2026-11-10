"""Tests for beat.verify: seal-manifest check, number registry and their use on this repository."""
import json
import subprocess

import pytest
import yaml

from beat import verify


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def commit(root, message):
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


# ---------------------------------------------------------------- rendering

@pytest.mark.parametrize("value, fmt, text", [
    (0.9942592, {"kind": "fixed", "digits": 3}, "0.994"),
    (0.474389, {"kind": "fixed", "digits": 1, "scale": 100}, "47.4"),
    (8.860481e-26, {"kind": "sci", "digits": 1}, "8.9 × 10⁻²⁶"),
    (8.860481e-26, {"kind": "sci_latex", "digits": 2}, "8.86 \\times 10^{-26}"),
    (9.96e-5, {"kind": "sci", "digits": 1}, "1.0 × 10⁻⁴"),
])
def test_render(value, fmt, text):
    assert verify.render(value, fmt) == text


# ---------------------------------------------------------------- numbers

@pytest.fixture
def number_repo(tmp_path):
    write(tmp_path / "results" / "t.csv", "kind,scenario,ba\nrep,main,0.99426\nrep,aux,0.52009\n")
    write(tmp_path / "results" / "d.json", json.dumps({"claims": {"C1": {"p": 3.2e-7}}}))
    write(tmp_path / "doc.md", "Within a day 0.994; across days 0.520. p = 3.2 × 10⁻⁷.\n")
    write(tmp_path / "demo.json", json.dumps({"v": [0.994259]}))
    registry = {"numbers": [
        {"id": "within", "text": "0.994", "format": {"kind": "fixed", "digits": 3},
         "value": {"file": "results/t.csv", "column": "ba", "where": {"scenario": "main"}},
         "in": {"doc.md": ["Within a day 0.994"]},
         "demo": [{"file": "demo.json", "json": "v.0", "tolerance": 1e-5}]},
        {"id": "cross", "text": "0.520", "format": {"kind": "fixed", "digits": 3},
         "value": {"file": "results/t.csv", "column": "ba", "agg": "min", "where": {"kind": "rep"}},
         "in": {"doc.md": ["across days 0.520"]}},
        {"id": "p", "text": "3.2 × 10⁻⁷", "format": {"kind": "sci", "digits": 1},
         "value": {"file": "results/d.json", "json": "claims.C1.p"},
         "in": {"doc.md": ["p = 3.2 × 10⁻⁷"]}},
    ]}
    write(tmp_path / "numbers.yaml", yaml.safe_dump(registry, allow_unicode=True))
    return tmp_path


def test_numbers_pass(number_repo):
    errors, count = verify.check_numbers(number_repo, "numbers.yaml")
    assert errors == [] and count == 4


def test_numbers_detect_changed_document(number_repo):
    write(number_repo / "doc.md", "Within a day 0.99; across days 0.520. p = 3.2 × 10⁻⁷.\n")
    errors, _ = verify.check_numbers(number_repo, "numbers.yaml")
    assert len(errors) == 1 and "lacks" in errors[0]


def test_numbers_detect_changed_result_and_demo(number_repo):
    write(number_repo / "results" / "t.csv", "kind,scenario,ba\nrep,main,0.98\nrep,aux,0.52009\n")
    errors, _ = verify.check_numbers(number_repo, "numbers.yaml")
    assert any("source gives 0.980" in e for e in errors)
    assert any("demo.json" in e for e in errors)


def test_numbers_require_a_unique_row(number_repo):
    registry = yaml.safe_load((number_repo / "numbers.yaml").read_text(encoding="utf-8"))
    del registry["numbers"][0]["value"]["where"]
    write(number_repo / "numbers.yaml", yaml.safe_dump(registry, allow_unicode=True))
    errors, _ = verify.check_numbers(number_repo, "numbers.yaml")
    assert any("2 rows, expected 1" in e for e in errors)


# ---------------------------------------------------------------- seals

@pytest.fixture
def seal_repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "core.autocrlf", "false")
    write(tmp_path / "beat" / "a.py", "x = 1\n")
    write(tmp_path / "configs" / "a.yaml", "k: 1\n")
    write(tmp_path / "results" / "sel.json", "{}\n")
    write(tmp_path / "tests" / "test_a.py", "pass\n")
    sealed = commit(tmp_path, "seal")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.org",
        "tag", "-a", "seal-1", "-m", "seal")
    manifest = {"scope": ["beat/*.py", "configs/*"],
                "seals": [{"name": "seal-1", "tag": "seal-1", "commit": sealed,
                           "selections": ["results/sel.json"]}],
                "sanctioned": []}
    write(tmp_path / "manifest.yaml", yaml.safe_dump(manifest))
    commit(tmp_path, "manifest")
    return tmp_path


def manifest_with(root, sanctioned):
    data = yaml.safe_load((root / "manifest.yaml").read_text())
    data["sanctioned"] = sanctioned
    write(root / "manifest.yaml", yaml.safe_dump(data))


def test_seals_pass_and_ignore_unsealed_paths(seal_repo):
    write(seal_repo / "tests" / "test_a.py", "assert True\n")
    write(seal_repo / "beat" / "new_module.py", "y = 2\n")  # not in the seal commit
    commit(seal_repo, "tests and a new module")
    assert verify.check_seals(seal_repo, "manifest.yaml") == []


def test_seals_detect_unsanctioned_commit(seal_repo):
    write(seal_repo / "results" / "sel.json", '{"changed": true}\n')
    commit(seal_repo, "edit selection")
    errors = verify.check_seals(seal_repo, "manifest.yaml")
    assert len(errors) == 2 and "unsanctioned change to results/sel.json" in errors[0]
    assert "sealed path results/sel.json at HEAD differs from" in errors[1]


def test_seals_accept_sanctioned_commit_for_listed_paths_only(seal_repo):
    write(seal_repo / "beat" / "a.py", "x = 2\n")
    write(seal_repo / "configs" / "a.yaml", "k: 2\n")
    change = commit(seal_repo, "deviation")
    manifest_with(seal_repo, [{"commit": change, "paths": ["beat/a.py"]}])
    commit(seal_repo, "sanction")
    errors = verify.check_seals(seal_repo, "manifest.yaml")
    assert len(errors) == 2
    assert errors[0] == f"seal-1: unsanctioned change to configs/a.yaml in {change[:7]}"
    assert "sealed path configs/a.yaml at HEAD differs from" in errors[1]


def test_seals_detect_change_made_in_a_merge_resolution(seal_repo):
    git(seal_repo, "checkout", "-q", "-b", "side")
    write(seal_repo / "tests" / "test_a.py", "assert 1\n")
    commit(seal_repo, "side")
    git(seal_repo, "checkout", "-q", "-")
    write(seal_repo / "tests" / "test_b.py", "assert 2\n")
    commit(seal_repo, "main")
    git(seal_repo, "-c", "user.name=t", "-c", "user.email=t@example.org",
        "merge", "-q", "--no-commit", "--no-ff", "side")
    write(seal_repo / "beat" / "a.py", "x = 3\n")
    commit(seal_repo, "merge with an edited sealed file")
    errors = verify.check_seals(seal_repo, "manifest.yaml")
    assert len(errors) == 1 and "sealed path beat/a.py at HEAD differs from" in errors[0]


def test_seals_detect_working_tree_line_endings(seal_repo):
    write(seal_repo / "configs" / "a.yaml", "k: 1\r\n")
    errors = verify.check_seals(seal_repo, "manifest.yaml")
    assert errors == ["sealed path configs/a.yaml differs from HEAD in the working tree "
                      "(content or line endings)"]


def test_seals_detect_wrong_tag(seal_repo):
    data = yaml.safe_load((seal_repo / "manifest.yaml").read_text())
    data["seals"][0]["commit"] = git(seal_repo, "rev-parse", "HEAD")
    write(seal_repo / "manifest.yaml", yaml.safe_dump(data))
    assert any("tag seal-1 is" in e for e in verify.check_seals(seal_repo, "manifest.yaml"))


# ---------------------------------------------------------------- this repository

def has_seal_history():
    try:
        git(verify.ROOT, "rev-parse", "--verify", "-q", "seal-1^{commit}")
        return git(verify.ROOT, "rev-parse", "--is-shallow-repository") == "false"
    except subprocess.CalledProcessError:
        return False


needs_history = pytest.mark.skipif(not has_seal_history(),
                                   reason="needs the full Git history and the seal tags")


@needs_history
def test_repository_seal_manifest_passes():
    assert verify.check_seals() == []


@needs_history
def test_deviation7_paths_are_the_derived_set():
    manifest = yaml.safe_load((verify.ROOT / verify.MANIFEST).read_text(encoding="utf-8"))
    entry = next(e for e in manifest["sanctioned"] if e["kind"] == "deviation")
    seal1 = next(s for s in manifest["seals"] if s["name"] == "seal-1")
    changed = set(git(verify.ROOT, "diff", "--name-only", seal1["commit"], entry["commit"])
                  .splitlines())
    sealed = verify.sealed_paths(verify.ROOT, manifest)["seal-1"]
    assert sorted(sealed & changed) == sorted(entry["paths"])


def test_repository_registered_numbers_match():
    errors, count = verify.check_numbers()
    assert errors == [] and count > 0
