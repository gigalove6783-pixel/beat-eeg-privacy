"""Sequential command evidence; each command updates a single predictive ratio."""

import numpy as np


def trajectory(reference, commands, cues, alpha):
    """Return cumulative log BF for each probe prefix (no repeated-window evidence)."""
    prior = np.asarray(alpha, dtype=float)
    counts = np.zeros_like(prior)
    cumulative, result = 0.0, []
    for command, cue in zip(commands, cues):
        same, different = prior[cue] + reference[cue] + counts[cue], prior[cue] + counts[cue]
        cumulative += np.log(same[command] / same.sum()) - np.log(
            different[command] / different.sum())
        result.append(cumulative)
        counts[cue, command] += 1
    return np.asarray(result)


def decision(path, alpha=0.05, beta=0.05):
    upper, lower = np.log((1 - beta) / alpha), np.log(beta / (1 - alpha))
    hit = np.flatnonzero((path >= upper) | (path <= lower))
    if not len(hit):
        return {"decision": "undecided", "commands": len(path)}
    t = int(hit[0])
    return {"decision": "link" if path[t] >= upper else "no link", "commands": t + 1}


def summarize(records):
    """Weighted stream denominators include undecided streams in FMR and FNMR."""
    output = {}
    all_decided, all_weight, lengths, length_weights = 0.0, 0.0, [], []
    for mated, name, error in ((True, "mated", "no link"), (False, "nonmated", "link")):
        subset = [r for r in records if r["mated"] == mated]
        total = sum(r["weight"] for r in subset)
        decided = sum(r["weight"] for r in subset if r["decision"] != "undecided")
        output[name] = {
            "streams": len(subset),
            "error_rate": sum(r["weight"] for r in subset if r["decision"] == error) / total,
            "undecided": 1 - decided / total,
        }
        all_decided += decided
        all_weight += total
        for record in subset:
            if record["decision"] != "undecided":
                lengths.append(record["commands"])
                length_weights.append(record["weight"])
    output["share_decided"] = all_decided / all_weight
    if lengths:
        order = np.argsort(lengths)
        w = np.cumsum(np.array(length_weights)[order])
        output["median_commands_to_decision"] = int(np.array(lengths)[order][
            np.searchsorted(w, w[-1] / 2)])
    else:
        output["median_commands_to_decision"] = None
    output["FMR"] = output["nonmated"]["error_rate"]
    output["FNMR"] = output["mated"]["error_rate"]
    output["nominal_alpha_beta"] = [0.05, 0.05]
    output["model_ville_bounds"] = [0.05 / 0.95, 0.05 / 0.95]
    output["interpretation"] = "Operating characteristics, not real-stream error guarantees"
    return output
