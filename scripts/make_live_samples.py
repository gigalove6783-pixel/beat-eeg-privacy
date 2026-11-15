"""Build the three input files of the BEAT Audit Live demo (demo/live).

Run from the repository root:

    .venv\\Scripts\\python scripts/make_live_samples.py

Two sets of the same three kinds are written:

- ``demo/live/samples/yang2025/`` (ignored by Git): a subset of the local Yang2025 exports
  (``artifacts/audit/inputs/yang2025``, format in ``results/audit/export_manifest.json``).
  A = full feature vectors (rung P0), B = decoded commands (rung P2-RC_d8, tau 0.60) of the
  same trials, joined on (user, source_session, source_run, source_trial) as the manifest
  requires, C = file A with its user labels permuted. The subset keeps the first USERS users
  in label order, the deployment days 2 and 3 and the first TRIALS trials of each segment.
  Values are copied exactly (shortest round-trip float text); nothing is transformed.
- ``demo/live/samples/synthetic/`` (tracked): generated data of the same three kinds, so that
  a fresh clone can run the demo without the cohort.

C keeps every row (outputs, cue, day, segment, order) and permutes only the user label within
each (day, segment, order) position, so every pseudo-user receives one trial of some real
user at each position. A manifest with the SHA-256 and a description of each file lets the
page recognise prepared files by content, never by file name.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "demo" / "live" / "samples"
EXPORTS = ROOT / "artifacts" / "audit" / "inputs" / "yang2025"
FEATURES = "features_P0_seed0"
COMMANDS = "commands_P2-RC_d8_tau0.60_seed0"
# Fixed before any result was seen: the lowest user identifiers, as many as keep a feature run
# within about 15 s on the demo machine but never below the tool's 20-user stability threshold.
USERS = 20
DAYS = (2, 3)
TRIALS = 50
PERMUTATION_SEED = 20261116
# The subset rule as shown on the card's small print and the print page.
SELECTION = {
    "yang2025": (f"부분집합 규칙(결과와 무관하게 미리 고정): 사용자 번호가 가장 작은 {USERS}명, "
                 f"{DAYS[0]}·{DAYS[1]}일째 기록, 구간마다 처음 {TRIALS}개 시행. A·B·C는 같은 "
                 f"사용자·날·시행이고, C는 A의 사용자 라벨을 고정 시드({PERMUTATION_SEED})로 섞은 것."),
    "synthetic": (f"합성 데이터: 사용자 {USERS}명, 2일, 구간마다 {TRIALS}개 시행(실제 코호트 표본과 "
                  f"같은 크기). C는 A의 사용자 라벨을 고정 시드로 섞은 것."),
}
JOIN = ("user", "source_session", "source_run", "source_trial")


def _write(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value):
    """Exact text of one stored value: integers as integers, floats by shortest round trip."""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return repr(float(value))


def _feature_rows(data, index, users):
    x = data["outputs"]
    return [[users[n], _text(data["day"][i]), _text(data["segment"][i]), _text(data["order"][i]),
             _text(data["target"][i]), *(_text(v) for v in x[i])] for n, i in enumerate(index)]


def _feature_columns(dimension):
    return ["user", "day", "segment", "order", "target",
            *(f"outputs_{j}" for j in range(dimension))]


def _permuted_users(user, day, segment, order, seed):
    """Permute user labels within each (day, segment, order) position; values stay in place."""
    rng = np.random.default_rng(seed)
    permuted = np.array(user, dtype=object)
    keys = {}
    for i, key in enumerate(zip(day, segment, order)):
        keys.setdefault(key, []).append(i)
    users = sorted(set(user))
    for key in sorted(keys):
        rows = np.array(keys[key])
        if len(rows) != len(users):
            raise ValueError(f"position {key}: every user needs one trial for the control")
        permuted[rows] = np.array(user, dtype=object)[rng.permutation(rows)]
    return permuted.tolist()


def _sidecar(stem):
    with (EXPORTS / f"{stem}.trials.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _subset(data):
    """First TRIALS trials (by order) of each (user, day, segment) for the first USERS users."""
    users = sorted(set(data["user"].tolist()))[:USERS]
    chosen = []
    for u in users:
        for d in DAYS:
            for s in sorted(set(data["segment"].tolist())):
                ix = np.flatnonzero((data["user"] == u) & (data["day"] == d) & (data["segment"] == s))
                ix = ix[np.argsort(data["order"][ix], kind="stable")]
                if len(ix) < TRIALS:
                    raise ValueError(f"user {u} day {d} segment {s}: only {len(ix)} trials")
                chosen.extend(ix[:TRIALS].tolist())
    return np.array(chosen)


def build_cohort():
    with np.load(EXPORTS / f"{FEATURES}.npz", allow_pickle=False) as archive:
        features = {key: archive[key] for key in archive.files}
    with np.load(EXPORTS / f"{COMMANDS}.npz", allow_pickle=False) as archive:
        commands = {key: archive[key] for key in archive.files}
    index = _subset(features)
    keys_f, keys_c = _sidecar(FEATURES), _sidecar(COMMANDS)
    position = {tuple(row[k] for k in JOIN): int(row["row"]) for row in keys_c}
    if len(position) != len(keys_c):
        raise ValueError("commands sidecar: join keys are not unique")
    command_index = np.array([position[tuple(keys_f[i][k] for k in JOIN)] for i in index])
    for name in ("user", "day", "target"):
        if not np.array_equal(features[name][index], commands[name][command_index]):
            raise ValueError(f"{name}: joined trials disagree between features and commands")
    labels = [str(int(u)) for u in features["user"][index]]
    out = SAMPLES / "yang2025"
    files = {}
    a = out / "A_yang2025_features.csv"
    columns = _feature_columns(features["outputs"].shape[1])
    files[a.name] = _write(a, columns, _feature_rows(features, index, labels))
    b = out / "B_yang2025_commands.csv"
    command_rows = [[labels[n], _text(commands["day"][i]), _text(commands["segment"][i]),
                     _text(commands["order"][i]), _text(commands["target"][i]),
                     _text(commands["C"].item()), _text(commands["outputs"][i])]
                    for n, i in enumerate(command_index)]
    files[b.name] = _write(b, ["user", "day", "segment", "order", "target", "C", "outputs"],
                           command_rows)
    permuted = _permuted_users(labels, features["day"][index].tolist(),
                               features["segment"][index].tolist(),
                               features["order"][index].tolist(), PERMUTATION_SEED)
    c = out / "C_yang2025_shuffled_users.csv"
    files[c.name] = _write(c, columns, _feature_rows(features, index, permuted))
    source = (f"Yang2025, users {labels[0]}-{labels[-1]} ({USERS}), days {DAYS[0]}-{DAYS[1]}, "
              f"first {TRIALS} trials per segment")
    return out, files, source, {
        "A": f"{FEATURES}.npz", "B": f"{COMMANDS}.npz (same trials as A)",
        "C": f"{FEATURES}.npz, user labels permuted within each (day, segment, order)",
    }


def build_synthetic(users=USERS, per_segment=TRIALS, dimension=16, seed=20261116):
    """Planted user means and day drift; commands are a fixed decoder of the same features."""
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=0.55, size=(users, dimension))
    # A personal bias and spread in the decoded direction, so commands keep a weaker trace of
    # the user (how often each person's trials end in left, right or hold for a given cue).
    means[:, 0] += rng.normal(scale=0.8, size=users)
    spread = np.ones((users, dimension))
    spread[:, :2] = rng.uniform(0.4, 1.6, size=(users, 1))
    cue = np.zeros(dimension)
    cue[:2] = (1.2, -1.2)
    rows, labels, meta = [], [], []
    for u in range(users):
        for d in (1, 2):
            drift = rng.normal(scale=0.25, size=dimension)
            for s in (1, 2):
                for t in range(per_segment):
                    target = int(rng.integers(2))
                    x = means[u] + drift + (cue if target else -cue)
                    x = x + spread[u] * rng.normal(size=dimension)
                    rows.append(x)
                    labels.append(f"p{u + 1:02}")
                    meta.append((d, s, (s - 1) * per_segment + t, target))
    x = np.array(rows)
    score = x[:, :2] @ np.array([1.0, -1.0]) / np.sqrt(2)
    command = np.where(score > 0.5, 2, np.where(score < -0.5, 1, 0))
    out = SAMPLES / "synthetic"
    files = {}
    columns = _feature_columns(dimension)

    def feature_rows(user):
        return [[user[i], d, s, o, t, *(_text(v) for v in x[i])]
                for i, (d, s, o, t) in enumerate(meta)]

    a = out / "A_synthetic_features.csv"
    files[a.name] = _write(a, columns, feature_rows(labels))
    b = out / "B_synthetic_commands.csv"
    files[b.name] = _write(b, ["user", "day", "segment", "order", "target", "C", "outputs"],
                           [[labels[i], d, s, o, t, 3, int(command[i])]
                            for i, (d, s, o, t) in enumerate(meta)])
    # Its own seed: reusing the generator's seed would replay the stream that drew the data.
    permuted = _permuted_users(labels, *zip(*[(d, s, o) for d, s, o, _ in meta]),
                               PERMUTATION_SEED + 1)
    c = out / "C_synthetic_shuffled_users.csv"
    files[c.name] = _write(c, columns, feature_rows(permuted))
    source = f"synthetic, {users} users, 2 days, {2 * per_segment} trials per day"
    return out, files, source, {
        "A": f"generated features ({dimension} dimensions)",
        "B": "commands decoded from file A's features by a fixed threshold rule",
        "C": "file A, user labels permuted within each (day, segment, order)",
    }


ROLES = {
    "A": ("특징 벡터", "BCI가 보내는 특징 벡터 전체"),
    "B": ("명령", "같은 기록에서 해독한 명령(왼쪽·오른쪽·정지)"),
    "C": ("대조군", "A의 사용자 라벨만 섞은 대조군"),
}


def write_manifest(directory, files, source, derivation):
    entries = []
    for name, sha in files.items():
        role = name.split("_", 1)[0]
        title, description = ROLES[role]
        entries.append({"file": name, "sha256": sha, "role": role, "title": title,
                        "description": description, "derivation": derivation[role],
                        "source": source, "selection": SELECTION[directory.name],
                        "tier": "시연용 부분집합, 탐색"})
    (directory / "manifest.json").write_text(
        json.dumps({"set": directory.name, "files": entries}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic-only", action="store_true",
                        help="skip the Yang2025 subset (no local cohort exports needed)")
    args = parser.parse_args(argv)
    built = [build_synthetic()]
    if not args.synthetic_only:
        built.append(build_cohort())
    for directory, files, source, derivation in built:
        write_manifest(directory, files, source, derivation)
        for name, sha in files.items():
            print(f"{directory.relative_to(ROOT) / name}  {sha}")


if __name__ == "__main__":
    main()
