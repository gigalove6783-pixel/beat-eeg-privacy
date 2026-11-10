"""Presentation figures rebuilt from the result tables: ``python -m beat.figures``.

The experiment modules keep their own diagnostic figures; these are the
summary figures used in the README and slides.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ROOT  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
IDENTITY_CHANCE = 1 / 9
COMMON, RC, GREY = "#1f77b4", "#d95f02", "#555555"
DECODERS = {"B1": "Band power + LR", "B2": "CSP + shrinkage LDA", "B3": "Riemannian tangent space + LR"}


def calibration_figure():
    scores = pd.read_csv(RESULTS / "exp_a" / "subject_scores.csv")
    h1 = json.loads((RESULTS / "exp_a" / "h1.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for offset, (decoder, label) in zip((-0.06, 0.0, 0.06), DECODERS.items()):
        by_runs = scores[scores.decoder == decoder].groupby("n_runs").ba
        mean, sd = by_runs.mean(), by_runs.std()
        ax.errorbar(mean.index + offset, mean.values, yerr=sd.values, marker="o", capsize=3, label=label)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set(xticks=[1, 2, 3, 6], xlabel="Calibration runs from session T (24 trials each)",
           ylabel="Balanced accuracy on session E", ylim=(0.45, 1.0),
           title="Experiment A: decoders vs calibration amount (9 subjects)")
    ax.text(1.05, 0.93, f"1 run, Riemannian − CSP: {100 * h1['mean_difference']:+.1f} pp\n"
                        f"exact sign-flip p = {h1['p_value']:.2f} (one-sided)", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.5, 0.005, "Points: mean over subjects; bars: SD across subjects.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def tradeoff_figure():
    t = pd.read_csv(RESULTS / "exp_b" / "tradeoff.csv")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    baselines = {
        "P0": ("s", GREY, True, "P0  253-d tangent vector"),
        "P0-RC": ("s", RC, False, "P0-RC  recentred 253-d"),
        "P2": ("o", COMMON, True, "P2  8-d task bottleneck"),
        "P2-RC": ("o", RC, True, "P2-RC  recentred + 8-d bottleneck"),
    }
    for condition, (marker, color, filled, label) in baselines.items():
        r = t[t.condition == condition].iloc[0]
        ax.errorbar(r.task_ba, r.identity_ba,
                    xerr=[[r.task_ba - r.task_low], [r.task_high - r.task_ba]],
                    yerr=[[r.identity_ba - r.identity_low], [r.identity_high - r.identity_ba]],
                    fmt=marker, ms=10, color=color, mfc=color if filled else "white", mew=2,
                    elinewidth=1, capsize=3, alpha=0.9, label=label, zorder=3)
    for condition, color in (("P3", COMMON), ("P3-RC", RC)):
        curve = t[t.condition == condition].sort_values("target")
        ax.plot(curve.task_ba, curve.identity_ba, "-^", color=color, ms=6, lw=1.5,
                label=f"{condition}  + identity adversary, λ 0.01→3", zorder=4)
        chosen = curve[curve.selected]
        ax.plot(chosen.task_ba, chosen.identity_ba, "*", color="black", ms=14, zorder=5)
    for _, r in t[(t.condition == "P3") & (t.target >= 0.1)].iterrows():
        ax.annotate(f"λ={r.target:g}", (r.task_ba, r.identity_ba), xytext=(8, -3),
                    textcoords="offset points", fontsize=8, color=COMMON)
    ax.plot([], [], "*", color="black", ms=12, label="λ selected on session T (0.01)")
    ax.axhline(IDENTITY_CHANCE, color=GREY, ls="--", lw=1)
    ax.text(0.845, IDENTITY_CHANCE + 0.015, "identity chance 1/9", fontsize=8, color=GREY, ha="right")
    ax.annotate("better", xy=(0.83, 0.2), xytext=(0.77, 0.34), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GREY), color=GREY)
    ax.set(xlim=(0.55, 0.85), ylim=(0, 1.05), xlabel="Left/right MI balanced accuracy (session E)",
           ylabel="Identity BA of the strongest independent attacker",
           title="Experiment B: utility vs identity exposure")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.text(0.5, 0.005, "Bars: 95% subject/seed bootstrap intervals. Attackers are retrained on frozen "
                         "outputs collected after deployment (E runs 1–3 ↔ 4–6).", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def observation_figure():
    s = pd.read_csv(RESULTS / "exp_c" / "observation_strongest.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for condition, color, label in (("P0", GREY, "P0 253-d"), ("P2", COMMON, "P2 8-d"),
                                    ("P2-RC", RC, "P2-RC recentred 8-d")):
        feature = s[(s.condition == condition) & (s["mode"] == "feature")].sort_values("k")
        ax.errorbar(feature.k, feature.ba_mean, yerr=[feature.ba_mean - feature.ba_low,
                                                      feature.ba_high - feature.ba_mean],
                    fmt="-o", color=color, capsize=3, label=f"{label}: features sent")
        command = s[(s.condition == condition) & (s["mode"] == "command")]
        band = command.groupby("k").ba_mean.agg(["min", "max"])
        no_hold = command[command.tau == 0.5].sort_values("k")
        ax.plot(no_hold.k, no_hold.ba_mean, "--s", color=color, ms=5,
                label=f"{label}: commands only")
        ax.fill_between(band.index, band["min"], band["max"], color=color, alpha=0.15, lw=0)
    ax.axhline(IDENTITY_CHANCE, color=GREY, ls=":", lw=1)
    ax.axhline(2 * IDENTITY_CHANCE, color="#c0392b", ls=":", lw=1)
    ax.text(0.88, IDENTITY_CHANCE - 0.035, "chance 1/9", fontsize=8, color=GREY)
    ax.text(0.88, 2 * IDENTITY_CHANCE + 0.012, "2 × chance", fontsize=8, color="#c0392b")
    ax.set(xscale="log", xlim=(0.85, 24), ylim=(0, 1.05),
           xlabel="Consecutive outputs observed by the receiver (k)",
           ylabel="Identity BA of the strongest attacker",
           title="Experiment C: identity leakage vs observation length")
    ax.set_xticks([1, 5, 10, 20], labels=["1", "5", "10", "20"])
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    fig.text(0.5, 0.005, "Command lines: τ = 0.5 (no holds); shaded: range over τ 0.5–0.8. "
                         "Bars: 95% run-within-subject bootstrap.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (("exp_a_calibration", calibration_figure), ("exp_b_tradeoff", tradeoff_figure),
                        ("exp_c_observation_length", observation_figure)):
        fig = build()
        fig.savefig(OUT / f"{name}.png", dpi=180)
        plt.close(fig)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
