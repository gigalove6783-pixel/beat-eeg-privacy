"""Portable JSON, plain-language Markdown, and noninteractive PNG reports."""

import csv
import json
from pathlib import Path

import numpy as np

from .io import digest
from .protocol import POST_HOC_RANKING, report_entries, resolved_pairs_from_report


def _safe(text):
    return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def write_report(report, out):
    """Write the report and four diagnostic PNGs without opening any UI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    entries = report_entries(report)
    report.setdefault("unavailable_analyses", entries[len(report["cells"]):])
    for cell in report["cells"]:
        if "strongest" in cell:
            cell.setdefault("strongest_interpretation", POST_HOC_RANKING)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Transmitted-output leakage audit", "", report["interpretation"], "",
        "## Same-day beside cross-day", "",
        "| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |",
        "|---|---:|---|---|---:|---|---:|",
    ]
    notes = []
    for cell in entries:
        prefix = f"| {_safe(cell['condition'])} | {cell['k']} |"
        if cell["status"] != "ok":
            lines.append(f"{prefix} — | {_safe(cell['status'] + ': ' + cell['reason'])} | — | — | — |")
            continue
        for name, value in cell["comparators"].items():
            interval = value["interval"]
            unreliable = " (unreliable: small folds)" if value["unreliable"] else ""
            lines.append(
                f"{prefix} {name} | {value['wording']} | {value['auc']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}]{unreliable} | {value['holm_p']:.5g} |")
        notes.append(f"- {_safe(cell['condition'])}, k={cell['k']}: "
                     f"excluded users {_safe(cell['excluded_users'])}; "
                     f"strongest comparator {cell['strongest']}: {POST_HOC_RANKING}.")
    lines += ["", *notes, "", "## Detection over k", "", str(report["smallest_detected_k"]), "",
              "Any detected condition remains detected regardless of another condition's result.",
              "", "## Limitations and reasonably likely means", "",
              "- Linkage tests the association of physical A/B blocks, conditional on fitted attacks.",
              "- EER and fixed-FPR TPR are descriptive; deployed thresholds use development users.",
              "- Estimated log LR depends on logistic calibration and transfer to new users/days.",
              "- Cllr_min is a target-fitted oracle diagnostic, never feedback to the attacker.",
              "- D_sys is linkability, not identification probability.",
              "- Gallery extrapolation is an independence sensitivity scenario, not population risk.",
              "- Sequential error guarantees apply only to its generative hypotheses; drift and "
              "dependence can violate them.",
              "- Comparator training uses no target users, but a BCI encoder may have seen them.",
              "- These metrics describe technical means an attacker might use; they do not decide "
              "whether means are reasonably likely under GDPR Recital 26 or any other law.",
              "- Not detected means insufficient evidence, not absence; never a safety claim.",
              "", "## Guards", "", *[f"- {_safe(w)}" for w in report["warnings"]]]
    for cell in report["cells"]:
        for name, value in cell["comparators"].items():
            if "warning" in value:
                lines.append(f"- {_safe(cell['condition'])}/{name}: {value['warning']}")
            if "gallery" in value:
                lines.append(f"- Gallery: {value['gallery']['interpretation']}")
    lines += ["", "## Provenance", "",
              f"Chronological day order: `{report.get('day_order')}`; "
              f"source: `{report.get('day_order_source', 'not recorded by legacy writer')}`.",
              f"Source SHA-256: `{report['hashes']['source']['sha256']}`",
              f"Configuration SHA-256: `{report['hashes']['configuration']}`",
              "Input and per-cell population hashes, exclusions and all numeric metrics are in "
              "`report.json`.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    rows = [(cell, name, value) for cell in report["cells"] if cell["status"] == "ok"
            for name, value in cell["comparators"].items()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for condition, name in sorted({(c["condition"], n) for c, n, _ in rows}):
        points = sorted((c["k"], v["auc"]) for c, n, v in rows
                        if c["condition"] == condition and n == name)
        ax.plot(*np.array(points).T, marker="o", label=f"{condition}/{name}")
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set(xlabel="Consecutive outputs k", ylabel="Mean per-user AUC", ylim=(0, 1))
    if rows:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "curve.png", dpi=120)
    plt.close(fig)
    for filename, xlabel, ylabel in (
        ("det", "False match rate (normal-deviate scale)",
         "False non-match rate (normal-deviate scale)"),
        ("scores", "Comparator score", "Weighted mass"),
        ("gallery", "Gallery size N", "Rank-1 probability"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for cell, name, value in rows:
            label = f"{cell['condition']}/k={cell['k']}/{name}"
            if filename == "det":
                from scipy.special import ndtri
                metric = value["metrics"]
                ax.plot(ndtri(np.clip(metric["fpr"], 1e-4, 1 - 1e-4)),
                        ndtri(np.clip(metric["fnmr"], 1e-4, 1 - 1e-4)), label=label)
            elif filename == "scores":
                h = value["histogram"]
                centers = (np.array(h["edges"][:-1]) + h["edges"][1:]) / 2
                ax.plot(centers, h["mated"], label=label + " mated")
                ax.plot(centers, h["nonmated"], "--", label=label + " nonmated")
            elif "gallery" in value:
                color = None
                for index, fold in enumerate(value["gallery"]["folds"]):
                    p = fold["check"]
                    line = ax.plot(
                        [r["N"] for r in p], [r["measured"] for r in p], "o-", alpha=0.4,
                        color=color, label=f"{label} held-out folds" if index == 0 else None,
                    )[0]
                    color = line.get_color()
                larger = value["gallery"]["larger_N"]
                if larger:
                    ax.plot([r["N"] for r in larger], [r["predicted"] for r in larger],
                            "--", color=color, label=label + " sensitivity only")
        ax.set(xlabel=xlabel, ylabel=ylabel)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=5, ncol=2)
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=120)
        plt.close(fig)


def compare_reports(reports, utilities, *, derived_from=None):
    """Merge alternatives only on identical cohort/days/physical A/B/k/populations.

    reports maps alternative names to report dicts or report.json paths. utilities
    maps those same names to task_ba, hold_rate, wrong_move_rate in [0,1].
    Actual analysed users, canonical population payload and resolved physical
    A/B selectors are checked. Other settings may legitimately differ.
    Cell-level detection and detecting comparators are separate from the
    target-observed (post hoc) highest-AUC comparator's descriptive statistics.
    Non-evaluable coverage rows remain visible; failed computations are refused.
    ``derived_from`` optionally maps command alternative -> source feature
    alternative. A higher observed command AUC flags an insufficient feature
    attacker; it never overwrites measured statistics or a detection.
    """
    signatures, rows, kinds = [], [], {}
    for name, item in reports.items():
        if isinstance(item, (str, Path)):
            item = json.loads(Path(item).read_text(encoding="utf-8"))
        if item.get("purpose") == "timing_only":
            raise ValueError("timing: run a complete audit before comparing alternatives")
        kinds[name] = item.get("kind")
        signature = []
        utility = utilities.get(name, {})
        for field in ("task_ba", "hold_rate", "wrong_move_rate"):
            if field not in utility or not np.isfinite(utility[field]) or not 0 <= utility[field] <= 1:
                raise ValueError(f"utility.{field}: supply a numeric rate in [0,1] for {name}")
        for cell in report_entries(item):
            if cell["status"] not in ("ok", "not_evaluable"):
                raise ValueError("status: fix failed computations before comparing reports")
            if cell.get("analysis_requested") is False:
                population_key = digest({"unavailable_population": cell["comparison_users"]})
            else:
                pop = cell["population"]
                if (sorted(cell["analysed_users"]) != sorted(pop["users"]) or
                        digest(pop) != cell["population_hash"]):
                    raise ValueError(
                        "population_hash: analysed users/masks/folds differ; rerun on one population")
                population_key = cell["population_hash"]
            physical_pairs = resolved_pairs_from_report(item, cell)
            signature.append((
                item["cohort"], tuple(item["days"]), cell["condition"], tuple(cell["days"]),
                cell["relation"], cell["k"], digest(physical_pairs), population_key,
            ))
            evaluable = cell["status"] == "ok"
            value = cell["comparators"][cell["strongest"]] if evaluable else None
            wording = cell["wording"] if evaluable else "not_evaluable"
            detected = sorted(name for name, result in cell["comparators"].items()
                              if evaluable and result["holm_p"] <= 0.05)
            rows.append({
                "alternative": name, **{k: utility[k] for k in
                                      ("task_ba", "hold_rate", "wrong_move_rate")},
                "cohort": item["cohort"], "condition": cell["condition"],
                "relation": cell["relation"], "k": cell["k"],
                "comparator": cell["strongest"] if evaluable else None,
                "auc": value["auc"] if evaluable else None,
                "interval_low": value["interval"][0] if evaluable else None,
                "interval_high": value["interval"][1] if evaluable else None,
                "wording": wording, "population_hash": cell["population_hash"],
                "status": cell["status"], "reason": cell.get("reason", ""),
                "cell_wording": wording,
                "selected_comparator_wording": value["wording"] if evaluable else "",
                "detected_comparators": ", ".join(detected),
                "comparator_selection": POST_HOC_RANKING if evaluable else "",
            })
        signatures.append(sorted(signature))
    if not signatures or any(s != signatures[0] for s in signatures[1:]):
        raise ValueError(
            "cohort/days/condition A/B selectors/k/population_hash: use reports with the same "
            "resolved physical conditions and analysed population")
    if derived_from:
        for row in rows:
            row.update(derived_from=derived_from.get(row["alternative"], ""),
                       data_processing_guard="")
        for command, feature in derived_from.items():
            if kinds.get(command) != "commands" or kinds.get(feature) != "features":
                raise ValueError("derived_from: map an included command alternative to its "
                                 "included source feature alternative")
            for crow in [r for r in rows if r["alternative"] == command and r["status"] == "ok"]:
                frow = next(r for r in rows if r["alternative"] == feature and
                            (r["condition"], r["relation"], r["k"]) ==
                            (crow["condition"], crow["relation"], crow["k"]))
                if frow["status"] == "ok" and crow["auc"] > frow["auc"]:
                    warning = (
                        f"Declared {command} derived from {feature}: feature attacker is too weak; "
                        f"the feature rung's exposure is at least the command rung's "
                        f"(observed command AUC {crow['auc']:.6g})."
                    )
                    for row in (crow, frow):
                        row["data_processing_guard"] = (
                            row["data_processing_guard"] + " " + warning).strip()
    return rows


def write_comparison(rows, out):
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
