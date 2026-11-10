"""Repository checks that need no EEG data: ``python -m beat.verify``.

(a) Seal manifest (``seals/manifest.yaml``): every commit after a seal that touched one of its
    sealed paths is a sanctioned commit, every sealed path at HEAD has the blob of the last seal
    or sanctioned commit covering it (this also catches merge resolutions), and the working tree
    equals HEAD byte for byte on every sealed path. Needs the full Git history and the seal tags.
(b) Number registry (``docs/numbers.yaml``): every registered headline number is recomputed from
    its result file, rendered as the documents print it, and found in each listed document
    snippet; demo-data values are compared with the result value.
(c) ``--references`` (network, run on request): every DOI, arXiv identifier and URL in the
    reference lists of README.md, docs/report.md and docs/related_work.md resolves.

Exit status 0 when every selected check passes.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("seals") / "manifest.yaml"
REGISTRY = Path("docs") / "numbers.yaml"
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


# ---------------------------------------------------------------- seals

def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True,
                          text=True, encoding="utf-8").stdout


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def blobs_at(root: Path, commit: str) -> dict[str, str]:
    """Blob id of every file in `commit`."""
    out = {}
    for line in git(root, "ls-tree", "-r", commit).splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def sealed_paths(root: Path, manifest: dict) -> dict[str, set[str]]:
    """Files of each seal commit that fall in the shared scope or the seal's selections."""
    out = {}
    for seal in manifest["seals"]:
        patterns = list(manifest["scope"]) + list(seal.get("selections", []))
        files = git(root, "ls-tree", "-r", "--name-only", seal["commit"]).splitlines()
        out[seal["name"]] = {f for f in files
                             if any(fnmatch.fnmatchcase(f, p) for p in patterns)}
    return out


def commits_after(root: Path, commit: str) -> list[tuple[str, list[str]]]:
    """Non-merge commits reachable from HEAD but not from `commit`, with their changed paths."""
    log = git(root, "log", "--no-merges", "--no-renames", "--format=%x00%H", "--name-only",
              f"{commit}..HEAD")
    out = []
    for block in log.split("\0")[1:]:
        lines = [line for line in block.splitlines() if line]
        out.append((lines[0], lines[1:]))
    return out


def check_seals(root: Path = ROOT, manifest_path: Path = MANIFEST) -> list[str]:
    manifest = yaml.safe_load((root / manifest_path).read_text(encoding="utf-8"))
    errors = []
    allowed = {}
    for entry in manifest.get("sanctioned", []):
        allowed.setdefault(entry["commit"], set()).update(entry["paths"])
    scopes = sealed_paths(root, manifest)
    for seal in manifest["seals"]:
        tagged = git(root, "rev-parse", f"{seal['tag']}^{{commit}}").strip()
        if tagged != seal["commit"]:
            errors.append(f"{seal['name']}: tag {seal['tag']} is {tagged}, manifest says "
                          f"{seal['commit']}")
        for commit, changed in commits_after(root, seal["commit"]):
            for path in sorted(scopes[seal["name"]].intersection(changed)):
                if path not in allowed.get(commit, ()):
                    errors.append(f"{seal['name']}: unsanctioned change to {path} in {commit[:7]}")
    head = blobs_at(root, "HEAD")
    # The commit walk skips merges, so also compare HEAD directly with the last commit that was
    # allowed to set each sealed path (its seal or a later sanctioned commit).
    order = {c: i for i, c in enumerate(git(root, "rev-list", "--topo-order", "HEAD").split())}
    trees = {}
    for path in sorted(set().union(*scopes.values())):
        target = root / path
        covering = [s["commit"] for s in manifest["seals"] if path in scopes[s["name"]]]
        covering += [c for c, paths in allowed.items() if path in paths]
        reachable = [c for c in covering if c in order]
        if not reachable:
            errors.append(f"no seal or sanctioned commit covering {path} is an ancestor of HEAD")
        elif path in head:
            last = min(reachable, key=order.__getitem__)
            if last not in trees:
                trees[last] = blobs_at(root, last)
            if trees[last].get(path) != head[path]:
                errors.append(f"sealed path {path} at HEAD differs from {last[:7]}, the last seal "
                              "or sanctioned commit covering it")
        if path not in head:
            errors.append(f"sealed path {path} is missing at HEAD")
        elif not target.is_file():
            errors.append(f"sealed path {path} is missing from the working tree")
        elif blob_sha1(target.read_bytes()) != head[path]:
            errors.append(f"sealed path {path} differs from HEAD in the working tree "
                          "(content or line endings)")
    return errors


# ---------------------------------------------------------------- numbers

def _norm(value):
    if isinstance(value, bool):
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def lookup(root: Path, spec: dict) -> float:
    """Value named by a registry `value` spec: a JSON path, or a CSV row filter and column."""
    path = root / spec["file"]
    if "json" in spec:
        node = json.loads(path.read_text(encoding="utf-8"))
        for key in str(spec["json"]).split("."):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return float(node)
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if all(_norm(r[k]) == _norm(v) for k, v in spec.get("where", {}).items())]
    values = [float(r[spec["column"]]) for r in rows]
    agg = spec.get("agg")
    if agg is None:
        if len(values) != 1:
            raise ValueError(f"{spec['file']} {spec.get('where')}: {len(values)} rows, expected 1")
        return values[0]
    if not values:
        raise ValueError(f"{spec['file']} {spec.get('where')}: no rows")
    return {"min": min, "max": max}[agg](values)


def render(value: float, fmt: dict) -> str:
    """Text of `value` as the documents print it."""
    kind, digits = fmt["kind"], int(fmt["digits"])
    scaled = value * float(fmt.get("scale", 1))
    if kind == "fixed":
        return f"{scaled:.{digits}f}"
    if kind in ("sci", "sci_latex"):
        exponent = math.floor(math.log10(abs(scaled)))
        mantissa = scaled / 10 ** exponent
        if round(mantissa, digits) >= 10:
            mantissa, exponent = mantissa / 10, exponent + 1
        if kind == "sci":
            return f"{mantissa:.{digits}f} × 10{str(exponent).translate(SUPERSCRIPT)}"
        return f"{mantissa:.{digits}f} \\times 10^{{{exponent}}}"
    raise ValueError(f"unknown format kind {kind}")


def check_numbers(root: Path = ROOT, registry_path: Path = REGISTRY) -> tuple[list[str], int]:
    registry = yaml.safe_load((root / registry_path).read_text(encoding="utf-8"))
    errors, texts = [], {}
    occurrences = 0
    for entry in registry["numbers"]:
        name = entry["id"]
        try:
            value = lookup(root, entry["value"])
        except (KeyError, IndexError, ValueError, OSError) as exc:
            errors.append(f"{name}: cannot read value ({exc})")
            continue
        text = render(value, entry["format"])
        if text != entry["text"]:
            errors.append(f"{name}: source gives {text}, registry says {entry['text']}")
        for doc, snippets in entry.get("in", {}).items():
            if doc not in texts:
                texts[doc] = (root / doc).read_text(encoding="utf-8")
            for snippet in snippets:
                occurrences += 1
                if entry["text"] not in snippet:
                    errors.append(f"{name}: snippet {snippet!r} does not contain {entry['text']}")
                elif snippet not in texts[doc]:
                    errors.append(f"{name}: {doc} lacks {snippet!r}")
        for spec in entry.get("demo", []):
            occurrences += 1
            try:
                shown = lookup(root, spec)
            except (KeyError, IndexError, ValueError, OSError) as exc:
                errors.append(f"{name}: cannot read demo value ({exc})")
                continue
            if abs(shown - value) > float(spec.get("tolerance", 1e-6)):
                errors.append(f"{name}: {spec['file']} {spec['json']} = {shown}, source {value}")
    return errors, occurrences


# ---------------------------------------------------------------- references (network)

REFERENCE_SECTIONS = {"README.md": ("## References", None),
                      "docs/report.md": ("## References", "## Appendix A"),
                      "docs/related_work.md": ("## References", "## Appendix")}
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s<>\"\])]+)")
ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
URL = re.compile(r"https?://[^\s<>\"\])]+")
USER_AGENT = "BEAT-reference-check/1.0 (https://github.com/gigalove6783-pixel/beat-eeg-privacy)"


def reference_targets(root: Path) -> list[tuple[str, str]]:
    """DOIs, arXiv identifiers and other URLs in the reference sections of the documents."""
    found = {}
    for doc, (start, stop) in REFERENCE_SECTIONS.items():
        text = (root / doc).read_text(encoding="utf-8")
        text = text[text.index(start):text.index(stop) if stop and stop in text else None]
        for url in URL.findall(text):
            url = url.rstrip(".,;")
            if "doi.org/" not in url and "arxiv.org/abs/" not in url:
                found.setdefault(("url", url), None)
        for doi in DOI.findall(text):
            found.setdefault(("doi", doi.rstrip(".,;")), None)
        for arxiv in ARXIV.findall(text) + re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", text):
            found.setdefault(("arxiv", arxiv), None)
    return sorted(found)


def _fetch(url: str, timeout: float = 30) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(200_000)
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def check_reference(kind: str, target: str) -> tuple[bool, str]:
    """Resolve one target; DOIs through the doi.org handle API, which publishers cannot block."""
    try:
        if kind == "doi":
            status, body = _fetch("https://doi.org/api/handles/" + urllib.parse.quote(target))
            data = json.loads(body or b"{}")
            urls = [v["data"]["value"] for v in data.get("values", []) if v.get("type") == "URL"]
            return data.get("responseCode") == 1 and bool(urls), (urls or [f"HTTP {status}"])[0]
        if kind == "arxiv":
            status, body = _fetch(f"http://export.arxiv.org/api/query?id_list={target}")
            title = re.search(rb"<entry>.*?<title>(.*?)</title>", body, re.S)
            return title is not None, (re.sub(r"\s+", " ", title.group(1).decode()) if title
                                       else f"HTTP {status}, no entry")
        status, _ = _fetch(target)
        return status < 400, f"HTTP {status}"
    except (OSError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


# Listed in docs/related_work.md as "not opened" (HTTP 403) and cited nowhere.
NOT_OPENED = {"https://ssrn.com/abstract=7459682"}


def check_references(root: Path = ROOT) -> list[str]:
    errors = []
    for kind, target in reference_targets(root):
        if target in NOT_OPENED:
            print(f"  skip {kind:5} {target} -> listed as not opened; not cited")
            continue
        ok, detail = check_reference(kind, target)
        print(f"  {'ok  ' if ok else 'FAIL'} {kind:5} {target} -> {detail}")
        if not ok:
            errors.append(f"{kind} {target}: {detail}")
    return errors


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m beat.verify", description=__doc__.split("\n")[0])
    parser.add_argument("--seals", action="store_true", help="run only the seal-manifest check")
    parser.add_argument("--numbers", action="store_true", help="run only the number check")
    parser.add_argument("--references", action="store_true",
                        help="only resolve every DOI, arXiv id and URL in the reference lists "
                             "(network; not part of the default run)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args(argv)
    if args.references:
        print(f"reference check, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
        errors = check_references(args.root)
        print(f"references: {'FAIL' if errors else 'pass'}")
        return 1 if errors else 0
    run_all = not (args.seals or args.numbers)
    failed = False
    if args.seals or run_all:
        errors = check_seals(args.root)
        print(f"seal manifest: {'FAIL' if errors else 'pass'}")
        failed |= bool(errors)
        for e in errors:
            print(f"  {e}")
    if args.numbers or run_all:
        errors, count = check_numbers(args.root)
        print(f"number registry: {'FAIL' if errors else 'pass'} ({count} occurrences checked)")
        failed |= bool(errors)
        for e in errors:
            print(f"  {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
