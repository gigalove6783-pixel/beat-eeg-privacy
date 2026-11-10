"""Summary figures for slides, README and the technical report.

Builds seven figures from committed result files only and writes PNG (200 dpi)
and SVG to ``results/figures/summary/``:

    python scripts/make_summary_figures.py

Inputs: results/exp_b, exp_c, exp_d, exp_e, exp_g/bnci and exp_l tables and the
seal records in docs/decision_log.md. The five-seed co-trained head score of
experiment L is recomputed from results/exp_l/head_predictions.csv.gz with the
project's own ``beat.exp_l.recalls`` and participant bootstrap ``interval``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beat.exp_l import interval, recalls  # noqa: E402
RES = ROOT / "results"
OUT = RES / "figures" / "summary"

CHANCE = 1 / 9

# Okabe-Ito colour-blind-safe palette, one meaning per colour across figures.
C = {
    "raw": "#000000",        # full tangent vector P0 / undefended
    "task": "#009E73",       # task-only bottleneck
    "grl": "#E69F00",        # gradient reversal (GRL-CE, EEGNet identifier)
    "maxent": "#CC79A7",     # maximum-entropy adversary
    "ensemble": "#56B4E9",   # ensemble-refresh adversary
    "head": "#0072B2",       # co-trained identity head / defended
    "attacker": "#D55E00",   # independently retrained attacker
    "chance": "#7F7F7F",
    "muted": "#5A5A5A",
    "pending": "#BDBDBD",
}
FIGSIZE = (12.8, 7.2)  # 16:9
TITLE_SIZE = 18
SUBTITLE_SIZE = 13
LABEL_SIZE = 13
TICK_SIZE = 11.5
NOTE_SIZE = 10.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": TICK_SIZE,
    "axes.titlesize": LABEL_SIZE,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": LABEL_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": TICK_SIZE,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "beat-summary",
})


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def new_figure(title, subtitle, tag, source, **subplot_kw):
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.02, 0.975, title, size=TITLE_SIZE, weight="bold", va="top")
    fig.text(0.02, 0.915, subtitle, size=SUBTITLE_SIZE, color=C["muted"], va="top")
    tag_colour = {"CONFIRMATORY": C["head"], "EXPLORATORY": C["grl"]}.get(tag, C["muted"])
    fig.text(0.985, 0.018, tag, size=NOTE_SIZE, weight="bold", color="white",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", fc=tag_colour, ec="none"))
    fig.text(0.02, 0.018, "Source: " + source, size=NOTE_SIZE - 1, color=C["muted"],
             va="bottom")
    return fig


def panel_tag(ax, text):
    """Status tag on the panel's title line, right-aligned above the axes."""
    colour = {"CONFIRMATORY": C["head"], "EXPLORATORY": C["grl"]}[text]
    ax.text(1.0, 1.035, text, transform=ax.transAxes, ha="right", va="bottom",
            size=NOTE_SIZE - 1.5, weight="bold", color=colour,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour, lw=1))


CHANCE_HANDLE = Line2D([], [], color=C["chance"], ls=":", lw=1.6, label="Chance (1/9)")


def chance_line(ax, label=True, x=0.01):
    ax.axhline(CHANCE, color=C["chance"], ls=":", lw=1.4, zorder=1)
    if label:
        ax.text(x, CHANCE + 0.012, "chance (1/9)", transform=ax.get_yaxis_transform(),
                size=NOTE_SIZE, color=C["chance"], va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, metadata={"Software": None})
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def lam_label(v):
    return f"{v:g}"


# ----------------------------------------------------------------------------
# Figure 1: co-trained head vs independent attackers
# ----------------------------------------------------------------------------
def load_pitfall1():
    """Return panel specs: list of dicts with series (variant, lambdas, head, independent)."""
    b = pd.read_csv(RES / "exp_b" / "adversary_comparison.csv")
    d = pd.read_csv(RES / "exp_d" / "tradeoff.csv")
    e = pd.read_csv(RES / "exp_e" / "lambda_summary.csv")

    def b_series(cond):
        s = b[b.condition == cond].sort_values("target")
        return dict(variant="grl", lam=s.target.tolist(), head=s.adversary_main_ba.tolist(),
                    indep=s.independent_ba.tolist())

    def d_series(cond, variant):
        s = d[d.condition == cond].sort_values("target")
        return dict(variant=variant, lam=s.target.tolist(), head=s.head_main_ba.tolist(),
                    indep=s.identity_ba.tolist())

    def e_series(alignment):
        s = e[e.alignment == alignment].sort_values("target")
        return dict(variant="grl", lam=s.target.tolist(), head=s.identifier_ba.tolist(),
                    indep=s.strongest_ba.tolist())

    return [
        dict(title="Common features", tag="CONFIRMATORY", series=[b_series("P3")]),
        dict(title="Common features", tag="EXPLORATORY",
             series=[d_series("MaxEnt", "maxent"), d_series("Ensemble", "ensemble")]),
        dict(title="No alignment", tag="EXPLORATORY", series=[e_series("none")]),
        dict(title="Recentred features", tag="CONFIRMATORY", series=[b_series("P3-RC")]),
        dict(title="Recentred features", tag="EXPLORATORY",
             series=[d_series("MaxEnt-RC", "maxent"), d_series("Ensemble-RC", "ensemble")]),
        dict(title="Euclidean alignment", tag="EXPLORATORY", series=[e_series("ea")]),
    ]


PITFALL1_COLUMNS = ["B · Gradient reversal", "D · MaxEnt and ensemble-refresh",
                    "E · EEGNet with co-trained identifier"]
VARIANT_MARKER = {"grl": "o", "maxent": "^", "ensemble": "s"}


def fig_pitfall1():
    panels = load_pitfall1()
    b = pd.read_csv(RES / "exp_b" / "adversary_comparison.csv")
    at1 = b[(b.condition == "P3") & (b.target == 1.0)].iloc[0]
    fig = new_figure(
        "Pitfall 1: the co-trained identity head says “private”; independent attackers disagree",
        f"Same frozen outputs, two judges. B at λ = 1: own head {at1.adversary_main_ba:.2f} "
        f"(below chance) vs retrained attacker {at1.independent_ba:.2f}.",
        "B CONFIRMATORY · D, E EXPLORATORY",
        "results/exp_b/adversary_comparison.csv, results/exp_d/tradeoff.csv, "
        "results/exp_e/lambda_summary.csv",
    )
    axes = fig.subplots(2, 3, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.765, bottom=0.215,
                        wspace=0.08, hspace=0.45)
    for ax, header in zip(axes[0], PITFALL1_COLUMNS):
        ax.text(0, 1.2, header, transform=ax.transAxes, size=LABEL_SIZE + 1.5,
                weight="bold", va="bottom")
    for ax, spec in zip(axes.flat, panels):
        lams = sorted({v for s in spec["series"] for v in s["lam"]})
        pos = {v: i for i, v in enumerate(lams)}
        n = len(spec["series"])
        offsets = [0.0] if n == 1 else [-0.17, 0.17]
        for s, off in zip(spec["series"], offsets):
            xs = [pos[v] + off for v in s["lam"]]
            m = VARIANT_MARKER[s["variant"]]
            ax.vlines(xs, s["head"], s["indep"], color="#C8C8C8", lw=3.2, zorder=2)
            ax.scatter(xs, s["head"], marker=m, s=46, facecolor="white",
                       edgecolor=C["head"], lw=1.8, zorder=3)
            ax.scatter(xs, s["indep"], marker=m, s=46, color=C["attacker"], zorder=3)
        ax.set_xticks(range(len(lams)), [lam_label(v) for v in lams])
        ax.set_xlim(-0.6, len(lams) - 0.4)
        ax.set_ylim(0, 1.05)
        ax.set_title(spec["title"], pad=7, weight="normal")
        chance_line(ax, label=False)
        panel_tag(ax, spec["tag"])
    for ax in axes[1]:
        ax.set_xlabel("Adversarial strength λ")
    for ax in axes[:, 0]:
        ax.set_ylabel("Identity balanced accuracy")

    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=C["head"], mew=1.8, ms=8,
               label="Co-trained identity head (the model's own judge)"),
        Line2D([], [], marker="o", ls="", color=C["attacker"], ms=8,
               label="Strongest independently retrained attacker"),
        Line2D([], [], color="#C8C8C8", lw=4, label="Gap = exposure the head misses"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=8,
               label="Gradient reversal (B, E)"),
        Line2D([], [], marker="^", ls="", mfc="none", mec="black", ms=8, label="MaxEnt (D)"),
        Line2D([], [], marker="s", ls="", mfc="none", mec="black", ms=8,
               label="Ensemble-refresh (D)"),
        CHANCE_HANDLE,
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.05),
               handletextpad=0.4, columnspacing=1.6)
    save(fig, "fig1_pitfall1_head_vs_attacker")


# ----------------------------------------------------------------------------
# Figure 2: utility-identity frontier
# ----------------------------------------------------------------------------
def load_frontier():
    b = pd.read_csv(RES / "exp_b" / "tradeoff.csv")
    d = pd.read_csv(RES / "exp_d" / "tradeoff.csv")
    out = {}
    for fam, b_cond, label in (("common", "P3", "Common features"),
                               ("rc", "P3-RC", "Recentred features (per-user offset removed)")):
        df = d[d.family == fam]
        task = df[df.kind == "task"].sort_values("d")
        out[fam] = dict(
            label=label,
            raw=df[df.kind == "linear"].iloc[0],
            task=task,
            adversarial={
                "grl": b[b.condition == b_cond].sort_values("target"),
                "maxent": df[df.kind == "maxent"].sort_values("target"),
                "ensemble": df[df.kind == "ensemble"].sort_values("target"),
            },
        )
    return out


D_LABEL_OFFSET = {("common", 32): (-9, -12, "right"), ("rc", 4): (9, 0, "left")}


def fig_frontier():
    data = load_frontier()
    d1 = {f: data[f]["task"].iloc[0] for f in data}
    fig = new_figure(
        "No adversarial defence beats a narrow task-only bottleneck",
        f"Lowest identity at full utility: a 1-d bottleneck ({d1['common'].identity_ba:.2f} "
        f"common, {d1['rc'].identity_ba:.2f} recentred) — still about twice chance.",
        "B GRL CONFIRMATORY · D EXPLORATORY",
        "results/exp_d/tradeoff.csv (task-only, MaxEnt, ensemble, P0), "
        "results/exp_b/tradeoff.csv (GRL)",
    )
    axes = fig.subplots(1, 2, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.235, wspace=0.07)
    for ax, fam in zip(axes, ("common", "rc")):
        spec = data[fam]
        task = spec["task"]
        ax.plot(task.task_ba, task.identity_ba, "-o", color=C["task"], lw=2.2, ms=7, zorder=4)
        for _, r in task.iterrows():
            # Labels sit left of the line; adversarial points cluster to its right.
            dx, dy, ha = D_LABEL_OFFSET.get((fam, int(r.d)), (-9, 0, "right"))
            ax.annotate(f"d = {int(r.d)}", (r.task_ba, r.identity_ba), xytext=(dx, dy),
                        textcoords="offset points", size=NOTE_SIZE, color=C["task"],
                        weight="bold", ha=ha, va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        raw = spec["raw"]
        ax.scatter(raw.task_ba, raw.identity_ba, marker="*", s=260, color=C["raw"], zorder=5)
        ax.annotate(f"{raw.condition}: full 253-d vector", (raw.task_ba, raw.identity_ba),
                    xytext=(10, -4), textcoords="offset points", size=NOTE_SIZE)
        for key, df in spec["adversarial"].items():
            ax.scatter(df.task_ba, df.identity_ba, marker=VARIANT_MARKER[key], s=58,
                       color=C[key], edgecolor="white", lw=0.6, zorder=3, alpha=0.95)
            sel = df[df.selected.astype(bool)]
            ax.scatter(sel.task_ba, sel.identity_ba, marker="o", s=230, facecolor="none",
                       edgecolor="black", lw=1.6, zorder=6)
        chance_line(ax)
        ax.set_title(spec["label"], pad=6)
        ax.set_xlabel("Task balanced accuracy (left vs right; higher = more useful)")
        ax.text(0.985, 0.04, "better: right and down →", transform=ax.transAxes,
                ha="right", size=NOTE_SIZE, color=C["muted"], style="italic")
    axes[0].set_ylabel("Identity balanced accuracy\n(strongest independent attacker)")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xlim(0.64, 0.735)
    handles = [
        Line2D([], [], marker="*", ls="", color=C["raw"], ms=13,
               label="No defence (P0 / P0-RC)"),
        Line2D([], [], marker="o", color=C["task"], lw=2.2, ms=7,
               label="Task-only bottleneck, width d"),
        Line2D([], [], marker="o", ls="", color=C["grl"], ms=8, label="Gradient reversal (B), all λ"),
        Line2D([], [], marker="^", ls="", color=C["maxent"], ms=8, label="MaxEnt (D), all λ"),
        Line2D([], [], marker="s", ls="", color=C["ensemble"], ms=8,
               label="Ensemble-refresh (D), all λ"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="black", mew=1.6, ms=13,
               label="λ selected on session T"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055),
               columnspacing=2.0)
    save(fig, "fig2_utility_identity_frontier")


# ----------------------------------------------------------------------------
# Figure 3: command accumulation
# ----------------------------------------------------------------------------
# Ordered so the three-column legend reads: common | recentred | references.
C_MODELS = [
    ("P0", "No defence (P0)", C["raw"], "-"),
    ("P2", "Bottleneck d = 8 (P2)", C["task"], "-"),
    ("P3", "GRL, T-selected λ (P3)", C["grl"], "-"),
    ("P0-RC", "No defence, recentred (P0-RC)", C["raw"], "--"),
    ("P2-RC", "Bottleneck, recentred (P2-RC)", C["task"], "--"),
    ("P3-RC", "GRL, recentred (P3-RC)", C["grl"], "--"),
]


def load_commands():
    s = pd.read_csv(RES / "exp_c" / "observation_strongest.csv")
    t = pd.read_csv(RES / "exp_c" / "observation_threshold.csv")
    feat = s[s["mode"] == "feature"]
    cmd = s[s["mode"] == "command"]
    band = cmd.groupby("k").ba_mean.agg(["min", "max"]).reset_index()
    cmd_t = t[t["mode"] == "command"]
    return dict(feature=feat, command=cmd[cmd.tau == 0.5], band=band,
                n_settings=len(cmd_t), n_by5=int((cmd_t.smallest_k <= 5).sum()))


def fig_commands():
    data = load_commands()
    fig = new_figure(
        "Even bare left / right / hold commands identify users as they accumulate",
        f"With 5 consecutive commands, {data['n_by5']} of {data['n_settings']} model × "
        "hold-threshold settings exceed twice chance (point estimates).",
        "CONFIRMATORY",
        "results/exp_c/observation_strongest.csv, results/exp_c/observation_threshold.csv",
    )
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.27, wspace=0.17)
    ks = [1, 5, 10, 20]
    for ax, key, title, ylim in (
        (axes[0], "feature", "Transmitting features", (0, 1.05)),
        (axes[1], "command", "Transmitting only commands (hold threshold τ = 0.5)", (0, 0.6)),
    ):
        df = data[key]
        if key == "command":
            band = data["band"]
            ax.fill_between(band.k, band["min"], band["max"], color="#D9D9D9", lw=0,
                            zorder=1)
        for cond, _, colour, ls in C_MODELS:
            r = df[df.condition == cond].sort_values("k")
            ax.plot(r.k, r.ba_mean, ls=ls, marker="o", color=colour, lw=2.2, ms=6, zorder=3)
        chance_line(ax, label=False)
        ax.axhline(2 * CHANCE, color=C["chance"], ls="--", lw=1.2, zorder=1)
        ax.set_xticks(ks)
        ax.set_xlim(0, 21)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=6)
        ax.set_xlabel("Consecutive outputs observed from one user (k)")
    axes[0].set_ylabel("Identity balanced accuracy\n(strongest attacker)")
    handles = [Line2D([], [], ls=ls, marker="o", color=c, lw=2.2, ms=6, label=lbl)
               for _, lbl, c, ls in C_MODELS]
    handles += [
        Line2D([], [], color="#D9D9D9", lw=9, label="Commands, range over all models and τ"),
        Line2D([], [], color=C["chance"], ls="--", lw=1.4, label="Twice chance (2/9)"),
        CHANCE_HANDLE,
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04),
               columnspacing=3.0, fontsize=NOTE_SIZE + 0.5, labelspacing=0.35)
    save(fig, "fig3_command_accumulation")


# ----------------------------------------------------------------------------
# Figure 4: Pitfall 2, experiment G
# ----------------------------------------------------------------------------
G_ATTACKERS = [("beat_logistic", "logistic regression", "-"),
               ("paper_ts_svm", "paper's tangent-space SVM", "--")]
G_N = ["1", "2", "5", "10", "20", "all"]


def load_enrolment():
    g = RES / "exp_g" / "bnci"
    enr = pd.read_csv(g / "enrolment_averaged.csv", dtype={"enrolment_n": str})
    enr = enr[(enr.k == 1) & (enr.status == "evaluable")]
    ident = pd.read_csv(g / "identity_averaged.csv")
    ref = ident[(ident.scenario == "reference_session") & (ident.k == 1)]
    paper = pd.read_csv(g / "paper_table1_comparison.csv")
    paper = paper[(paper.condition == "defended") & (paper.direction == "S1_to_S2")
                  & (paper.attacker == "paper_ts_svm")].iloc[0]
    return enr, ref, paper


def fig_enrolment():
    enr, ref, paper = load_enrolment()
    n1 = enr[(enr.condition == "defended") & (enr.attacker == "beat_logistic")
             & (enr.enrolment_n == "1")].balanced_accuracy.iloc[0]
    n5 = enr[(enr.condition == "defended") & (enr.attacker == "beat_logistic")
             & (enr.enrolment_n == "5")].balanced_accuracy.iloc[0]
    fig = new_figure(
        "Pitfall 2: a defence tested only against reference-session attackers",
        f"The published defence reproduces (11.1%), but one labelled deployment trial per user "
        f"restores {n1:.0%} identification, five restore {n5:.0%}.",
        "EXPLORATORY",
        "results/exp_g/bnci/enrolment_averaged.csv, identity_averaged.csv, "
        "paper_table1_comparison.csv",
    )
    ax = fig.subplots()
    fig.subplots_adjust(left=0.085, right=0.70, top=0.83, bottom=0.22)
    xs = list(range(1, len(G_N) + 1))
    # Reference-session points sit left of x = 0 so the paper's star does not hide them.
    ref_x = -0.12
    for cond, colour in (("undefended", C["raw"]), ("defended", C["head"])):
        for att, _, ls in G_ATTACKERS:
            r = enr[(enr.condition == cond) & (enr.attacker == att)].set_index("enrolment_n")
            ax.plot(xs, r.loc[G_N, "balanced_accuracy"], ls=ls, marker="o", color=colour,
                    lw=2.4, ms=7, zorder=3)
            ref_v = ref[(ref.condition == cond) & (ref.attacker == att)].balanced_accuracy.iloc[0]
            ax.scatter(ref_x, ref_v, marker="o", s=52, color=colour, zorder=3)
    star_x, star_y = 0.15, paper.paper_accuracy_percent / 100
    ax.scatter(star_x, star_y, marker="*", s=320, color=C["grl"], edgecolor="black", lw=0.8,
               zorder=5)
    ax.annotate(f"Paper's protocol: {paper.paper_accuracy_percent:.1f}% reported,\n"
                f"{paper.observed_accuracy_percent:.1f}% reproduced here (session 1 → 2)",
                (star_x, star_y), xytext=(0.75, 0.36), textcoords="data",
                size=NOTE_SIZE + 0.5, va="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1, shrinkB=9))
    ax.axvspan(-0.5, 0.5, color="#F2F2F2", zorder=0)
    chance_line(ax, x=0.83)
    ax.set_xticks([0] + xs, ["0\n(reference\nsession only)"] + G_N[:-1] + ["all"])
    ax.set_xlim(-0.5, len(G_N) + 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Labelled deployment-period trials per user available to the attacker")
    ax.set_ylabel("Users correctly identified\n(balanced accuracy, single trial)")

    handles = [
        Line2D([], [], color=C["raw"], lw=2.4, marker="o", label="Undefended data"),
        Line2D([], [], color=C["head"], lw=2.4, marker="o",
               label="Defended data\n(transform fitted on reference session)"),
    ] + [Line2D([], [], color="black", ls=ls, lw=2, label=f"Attacker: {lbl}")
         for _, lbl, ls in G_ATTACKERS] + [
        Line2D([], [], marker="*", ls="", ms=16, mfc=C["grl"], mec="black",
               label="Paper's reported result"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.715, 0.53),
               labelspacing=1.1)
    fig.text(0.715, 0.20, "BCI IV 2a, 9 users; both session\ndirections averaged; "
             "RBF-SVM and MLP\nattackers need ≥ 3 labels (not shown).",
             size=NOTE_SIZE, color=C["muted"], va="bottom")
    save(fig, "fig4_pitfall2_enrolment")


# ----------------------------------------------------------------------------
# Figure 5: pre-registration timeline
# ----------------------------------------------------------------------------
def load_seals():
    """Parse seal commits, dates and test counts from docs/decision_log.md."""
    text = (ROOT / "docs" / "decision_log.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", line)]

    def find(pattern):
        for row in rows:
            m = re.search(pattern, row)
            if m:
                return row[2:12], m
        return None, None

    seal1_date, m1 = find(r"\*\*E seal\*\*: commit `([0-9a-f]{7})[0-9a-f]*`.*?All (\d+) tests passed")
    seal2_date, m2 = find(r"\*\*E seal 2\*\*: commit `([0-9a-f]{7})[0-9a-f]*`.*?(\d+) tests passed")
    ab_date, m_ab = find(r"A and B final runs completed at commit `([0-9a-f]{7})`")
    g_date, _ = find(r"Experiment G added")
    _, m3_tests = find(r"\*\*Seal 3\*\*.*?(\d+) tests passed")
    seal3_date, m3 = find(r"Seal 3 commit: `([0-9a-f]{7})[0-9a-f]*`")
    _, m_l1 = find(r"Final attempt 1 of L .*?stopped .*?sealed guard `MemoryError")
    _, m_l2 = find(r"Final attempt 2 of L completed")
    if not (m1 and m2 and m_ab and g_date and m3 and m3_tests):
        raise ValueError("decision_log.md: expected seal/result entries not found")
    return dict(
        seal1=(m1.group(1), seal1_date, m1.group(2)),
        seal2=(m2.group(1), seal2_date, m2.group(2)),
        ab=(m_ab.group(1), ab_date),
        g=g_date,
        seal3=(m3.group(1), seal3_date, m3_tests.group(1)),
        l_rerun=bool(m_l1 and m_l2),
    )


def fig_timeline():
    s = load_seals()
    decisions = json.loads((RES / "exp_l" / "decision.json").read_text())["claims"]
    supported = [c for c, v in decisions.items() if v["decision"] == "supported"]
    rejected = [c for c, v in decisions.items() if v["decision"] == "not_supported"]
    fig = new_figure(
        "Confirmatory claims are sealed before their evaluation data are opened",
        "Order of seals and analyses in docs/decision_log.md; exploratory work reuses data "
        "already seen and makes no confirmatory claim.",
        "CONFIRMATORY + EXPLORATORY",
        "docs/decision_log.md (commits, dates, test counts, run attempts); "
        "L decisions from results/exp_l/decision.json",
    )
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.78])
    ax.set_xlim(-0.6, 7.0)
    ax.set_ylim(-2.0, 1.05)
    ax.axis("off")
    ax.annotate("", xy=(6.98, 0), xytext=(-0.55, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6), zorder=1)

    seals = [
        (0, "E seal 1", f"{s['seal1'][0]} · {s['seal1'][1]}\nplan v1 · {s['seal1'][2]} tests passed"),
        (2, "E seal 2", f"{s['seal2'][0]} · {s['seal2'][1]}\nplan v2 · {s['seal2'][2]} tests passed"),
        (5.3, "Seal 3", f"{s['seal3'][0]} · {s['seal3'][1]}\nplan v3 · {s['seal3'][2]} tests passed"),
    ]
    for x, name, detail in seals:
        ax.scatter(x, 0, marker="D", s=230, color="black", lw=2, zorder=3)
        ax.text(x, 0.62, name, ha="center", va="bottom", size=LABEL_SIZE + 1, weight="bold")
        ax.text(x, 0.56, detail, ha="center", va="top", size=NOTE_SIZE + 0.5, linespacing=1.4)

    status_colour = {"Confirmatory": C["head"], "Exploratory": C["grl"]}
    analyses = [
        (1, "A · B · C", "Confirmatory",
         f"A, B final at {s['ab'][0]};\nH1, H2 not supported;\ncommands leak\nabove chance"),
        (3, "D · E", "Exploratory",
         "reuse session E;\nno defence passes;\nhead–attacker\ngap recurs"),
        (4.2, "G · published\ndefence", "Exploratory",
         "no seal; paper's\noperating point;\nre-identified with\n1–5 labels"),
        (6.4, "L · Lee2019", "Confirmatory",
         f"54 users; {', '.join(supported)}\nsupported; {', '.join(rejected)} not\n"
         "supported (Holm)"),
    ]
    box_w, box_h, top = 1.05, 1.56, -0.3
    for x, name, status, detail in analyses:
        colour = status_colour[status]
        ax.plot([x, x], [0, top], color=colour, lw=1.5, zorder=1)
        ax.scatter(x, 0, s=70, color=colour, zorder=3)
        ax.add_patch(FancyBboxPatch((x - box_w / 2, top - box_h), box_w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec=colour, lw=2.2, zorder=2))
        ax.text(x, top - 0.1, name, ha="center", va="top", size=LABEL_SIZE,
                weight="bold", zorder=3, linespacing=1.15)
        ax.text(x, top - 0.55, status.upper(), ha="center", va="top", size=NOTE_SIZE - 0.5,
                weight="bold", color=colour, zorder=3)
        ax.text(x, top - 0.86, detail, ha="center", va="top", size=NOTE_SIZE,
                color="#333333", zorder=3, linespacing=1.3)
    if s["l_rerun"]:
        ax.text(6.4, top - box_h + 0.07, "run 1 stopped by sealed\nmemory guard; unchanged\n"
                "re-run", ha="center", va="bottom", size=NOTE_SIZE - 1.5, color=C["muted"],
                style="italic", zorder=3, linespacing=1.2)
    save(fig, "fig5_preregistration_timeline")


# ----------------------------------------------------------------------------
# Figure 6: experiment L, confirmatory claims C1-C4 (Lee2019/OpenBMI, 54 users)
# ----------------------------------------------------------------------------
L_N = 54
L_CHANCE = 1 / L_N
# Pre-registered plan-v3 thresholds (docs/report.md §6.7).
C1_MIN_GAP, C2_MIN_GAP, C2_TASK_MARGIN = 0.20, 0.20, -0.05
C3_FLOOR, C4_FLOOR = 2 / L_N, 10 / L_N
ATTACKER_NAME = {"logistic": "logistic regression", "beat_logistic": "logistic regression",
                 "rbf_svm": "RBF-SVM", "beat_rbf_svm": "RBF-SVM", "mlp": "MLP",
                 "beat_mlp": "MLP", "paper_ts_svm": "paper's tangent-space SVM"}


def strongest(ident, **sel):
    """The single sealed-maximum row of identity_summary.csv for one cell."""
    rows = ident[ident.strongest]
    for col, val in sel.items():
        rows = rows[rows[col] == val]
    if len(rows) != 1:
        raise ValueError(f"identity_summary.csv: expected one strongest row for {sel}")
    return rows.iloc[0]


def load_lee2019():
    L = RES / "exp_l"
    decision = json.loads((L / "decision.json").read_text())["claims"]
    ident = pd.read_csv(L / "identity_summary.csv", dtype={"enrolment": str})
    task = pd.read_csv(L / "task_summary.csv").set_index("key")["mean"]
    repro = json.loads((L / "reproduction.json").read_text())
    rep = dict(kind="representation", mode="feature", enrolled=L_N)
    c1_att = strongest(ident, key="P3_lambda1", **rep)
    # Five-seed head score on the same seeds as the confirmatory attacker row; the
    # three-seed head_vs_independent.csv pair is descriptive only.
    seeds = [int(v) for v in c1_att.seeds.split(",")]
    heads = pd.read_csv(L / "head_predictions.csv.gz")
    heads = heads[(heads.key == "P3_lambda1") & heads.seed.isin(seeds)]
    cfg = yaml.safe_load((ROOT / "configs" / "exp_l.yaml").read_text())
    head_ci = interval(heads, cfg, sorted(cfg["subjects"]), seeds,
                       sorted(heads.direction.unique()))
    head = dict(ba=float(recalls(heads).mean()), ci_low=head_ci[0], ci_high=head_ci[1])
    obs = dict(kind="observation", key="P2-RC_d8", mode="command", tau=0.5)
    return dict(
        decision=decision, task=task, repro=repro, head=head, c1_att=c1_att,
        p0=strongest(ident, key="P0", **rep), p2rc=strongest(ident, key="P2-RC_d8", **rep),
        c3=[strongest(ident, k=k, **obs) for k in (1, 20)],
        c4_ref=strongest(ident, kind="benchmark", key="G"),
        c4=[strongest(ident, kind="enrolment", key="G", enrolment=n) for n in ("1", "2", "5")],
    )


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant} × 10$^{{{int(exp)}}}$"


def decision_badge(ax, claim):
    supported = claim["decision"] == "supported"
    text = ("SUPPORTED" if supported else "NOT SUPPORTED") + f"  ·  Holm p = {fmt_p(claim['holm_p'])}"
    ax.text(1.0, 1.04, text, transform=ax.transAxes, ha="right", va="bottom",
            size=NOTE_SIZE - 0.5, weight="bold", color="white" if supported else "black",
            bbox=dict(boxstyle="round,pad=0.3", fc="#222222" if supported else "white",
                      ec="#222222", lw=1.2))


def interval_rows(ax, rows, xmax):
    """Rows top to bottom: (label, value, low, high, colour). Label sits above its interval."""
    n = len(rows)
    for i, (label, v, lo, hi, colour) in enumerate(rows):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=colour, lw=3.2, alpha=0.45, solid_capstyle="butt",
                zorder=3)
        ax.scatter(v, y, s=60, color=colour, zorder=4, clip_on=False)
        right = hi + 0.012 * xmax < 0.86 * xmax
        ax.text(hi + 0.012 * xmax if right else lo - 0.012 * xmax, y, f"{v:.3f}",
                ha="left" if right else "right", va="center", size=NOTE_SIZE, weight="bold",
                color=colour, zorder=4)
        ax.text(0.004 * xmax, y + 0.4, label, ha="left", va="bottom", size=NOTE_SIZE,
                color="#333333", zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_ylim(-0.55, n - 0.05)
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.axvline(L_CHANCE, color=C["chance"], ls=":", lw=1.6, zorder=1)


def threshold_line(ax, x):
    ax.axvline(x, color="black", ls="-.", lw=1.3, zorder=2)


def pp(x):
    return f"{x * 100:+.1f}".replace("-", "−") + " pp"


def fig_lee2019():
    d = load_lee2019()
    dec = d["decision"]
    ok = sum(v["decision"] == "supported" for v in dec.values())
    fig = new_figure(
        f"Confirmatory replication on 54 users: {ok} of {len(dec)} pre-registered claims hold",
        "Lee2019/OpenBMI, five seeds, Holm at α = 0.05. Both pitfalls and the bottleneck result "
        "replicate; command accumulation does not pass correction.",
        "CONFIRMATORY",
        "results/exp_l/decision.json, identity_summary.csv, task_summary.csv, "
        "head_predictions.csv.gz, reproduction.json",
    )
    axes = fig.subplots(2, 2, gridspec_kw=dict(height_ratios=[1, 1.3]))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.80, bottom=0.175, wspace=0.10,
                        hspace=0.5)
    (a1, a2), (a3, a4) = axes

    # C1: co-trained head vs strongest independent attacker (P3, λ = 1).
    h, att = d["head"], d["c1_att"]
    interval_rows(a1, [
        ("Co-trained identity head (gradient reversal, λ = 1)", h["ba"], h["ci_low"],
         h["ci_high"], C["head"]),
        (f"Strongest independent attacker ({ATTACKER_NAME[att.attacker]}), same outputs",
         att.ba, att.ci_low, att.ci_high, C["attacker"]),
    ], 1.0)
    a1.text(0.985, 0.5, f"gap {pp(att.ba - h['ba'])}\n"
            f"required ≥ {C1_MIN_GAP * 100:.0f} pp", transform=a1.transAxes, ha="right",
            va="center", size=NOTE_SIZE + 0.5, weight="bold")
    a1.set_title("C1 · Pitfall 1: head vs attacker", size=LABEL_SIZE - 0.5)
    decision_badge(a1, dec["C1"])

    # C2: recentred bottleneck vs no defence, with task non-inferiority.
    p0, p2 = d["p0"], d["p2rc"]
    util = dec["C2"]["utility"]
    interval_rows(a2, [
        (f"No defence (P0) · task BA {d['task']['P0']:.3f}", p0.ba, p0.ci_low, p0.ci_high,
         C["raw"]),
        (f"Recentred bottleneck d = 8 (P2-RC) · task BA {d['task']['P2-RC_d8']:.3f}", p2.ba,
         p2.ci_low, p2.ci_high, C["task"]),
    ], 1.0)
    a2.text(0.985, 0.04, f"identity {pp(p2.ba - p0.ba)} (required ≤ −{C2_MIN_GAP * 100:.0f} pp)\n"
            f"task {pp(util['mean'])}, lower bound {pp(util['lower'])}\n"
            f"(non-inferiority margin {pp(C2_TASK_MARGIN)})",
            transform=a2.transAxes, ha="right", va="bottom", size=NOTE_SIZE, weight="bold",
            linespacing=1.3)
    a2.set_title("C2 · Bottleneck vs no defence", size=LABEL_SIZE - 0.5)
    decision_badge(a2, dec["C2"])

    # C3: commands only, P2-RC d = 8, τ = 0.5, k = 1 vs k = 20.
    k1, k20 = d["c3"]
    interval_rows(a3, [
        ("1 command observed", k1.ba, k1.ci_low, k1.ci_high, C["task"]),
        ("20 consecutive commands observed", k20.ba, k20.ci_low, k20.ci_high, C["task"]),
    ], 0.25)
    threshold_line(a3, C3_FLOOR)
    a3.text(0.985, 0.5, "k = 20 exceeds k = 1, but its\nmargin over 2/54 is not\n"
            "significant after correction", transform=a3.transAxes, ha="right", va="center",
            size=NOTE_SIZE, weight="bold", linespacing=1.3)
    a3.set_title("C3 · Commands only (P2-RC, τ = 0.5)", size=LABEL_SIZE - 0.5)
    decision_badge(a3, dec["C3"])

    # C4: published defence (G) vs deployment enrolment, with the reference-session gate.
    ref = d["c4_ref"]
    rows = [(f"Reference-session attacker ({ATTACKER_NAME[ref.attacker]}): "
             f"{ref.ba:.2%}; paper {d['repro']['paper_accuracy']:.1%}", ref.ba, ref.ci_low,
             ref.ci_high, C["head"])]
    for r, n in zip(d["c4"], ("1", "2", "5")):
        label = f"{n} labelled deployment trial{'s' if n != '1' else ''} per user"
        rows.append((label + (" (tested)" if n == "5" else ""), r.ba, r.ci_low, r.ci_high,
                     C["head"]))
    interval_rows(a4, rows, 1.0)
    a4.scatter(d["repro"]["paper_accuracy"], len(rows) - 1, marker="*", s=230,
               color=C["grl"], edgecolor="black", lw=0.7, zorder=5)
    threshold_line(a4, C4_FLOOR)
    a4.set_title("C4 · Pitfall 2: published defence", size=LABEL_SIZE - 0.5)
    decision_badge(a4, dec["C4"])

    for ax in (a3, a4):
        ax.set_xlabel("Identity balanced accuracy (strongest attacker)", size=LABEL_SIZE - 1)
    handles = [
        Line2D([], [], color=C["muted"], lw=3.2, alpha=0.45, marker="o", mfc=C["muted"],
               mec=C["muted"], label="Estimate with 95% participant-bootstrap interval"),
        Line2D([], [], color=C["chance"], ls=":", lw=1.6, label="Chance (1/54)"),
        Line2D([], [], color="black", ls="-.", lw=1.3,
               label="Pre-registered floor (C3: 2/54, C4: 10/54)"),
        Line2D([], [], marker="*", ls="", ms=14, mfc=C["grl"], mec="black",
               label="Paper's reported result"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.045),
               columnspacing=1.6, fontsize=NOTE_SIZE + 0.5)
    save(fig, "fig6_lee2019_confirmatory")


# ----------------------------------------------------------------------------
# Figure 7: identity vs number of enrolled users (L, descriptive) and 2a reference
# ----------------------------------------------------------------------------
L_ENROLLED = [9, 18, 27, 54]
N_SERIES = [("P0", "P0", "No defence (P0)", C["raw"]),
            ("P2-RC_d8", "P2-RC", "Recentred bottleneck d = 8 (P2-RC)", C["task"])]


def load_crowd():
    ident = pd.read_csv(RES / "exp_l" / "identity_summary.csv")
    rows = ident[(ident.kind == "enrolled") & ident.strongest]
    two_a = pd.read_csv(RES / "exp_b" / "tradeoff.csv").set_index("condition")
    return rows, two_a


def fig_crowd():
    rows, two_a = load_crowd()
    series = {key: rows[rows.key == key].set_index("enrolled").loc[L_ENROLLED]
              for key, *_ in N_SERIES}
    p2 = series["P2-RC_d8"]
    fig = new_figure(
        "More users lower identity accuracy, but leakage relative to chance grows",
        f"Recentred bottleneck: {p2.ba[9]:.2f} with 9 users ({p2.ba[9] * 9:.1f}× chance), "
        f"{p2.ba[54]:.2f} with 54 users ({p2.ba[54] * 54:.1f}× chance); "
        f"BCI IV 2a's 9 users: {two_a.identity_ba['P2-RC']:.2f}.",
        "DESCRIPTIVE",
        "results/exp_l/identity_summary.csv (enrolled subsets, seeds 0–2); "
        "results/exp_b/tradeoff.csv (BCI IV 2a)",
    )
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.25, wspace=0.2)
    xs = pd.Series(L_ENROLLED, dtype=float)
    for ax, scale in zip(axes, ("ba", "ratio")):
        for key, two_a_key, _, colour in N_SERIES:
            s = series[key]
            f = xs.values if scale == "ratio" else 1.0
            ax.fill_between(L_ENROLLED, s.ci_low * f, s.ci_high * f, color=colour, alpha=0.15,
                            lw=0, zorder=2)
            ax.plot(L_ENROLLED, s.ba * f, color=colour, lw=2.4, marker="o", ms=7, zorder=3)
            v = two_a.identity_ba[two_a_key] * (9 if scale == "ratio" else 1)
            ax.scatter(9, v, marker="D", s=70, facecolor="white", edgecolor=colour, lw=2,
                       zorder=4)
        ax.set_xticks(L_ENROLLED)
        ax.set_xlim(5, 58)
        ax.set_xlabel("Users the attacker must tell apart (N)")
    a, b = axes
    grid = pd.Series(range(5, 59), dtype=float)
    a.plot(grid, 1 / grid, color=C["chance"], ls=":", lw=1.6, zorder=1)
    a.text(40, 1 / 40 + 0.015, "chance (1/N)", size=NOTE_SIZE, color=C["chance"])
    a.set_ylim(0, 1.05)
    a.set_ylabel("Identity balanced accuracy\n(strongest attacker)")
    a.set_title("Accuracy")
    b.axhline(1, color=C["chance"], ls=":", lw=1.6, zorder=1)
    b.text(56, 1.6, "chance", size=NOTE_SIZE, color=C["chance"], ha="right")
    b.set_ylim(0, 60)
    b.set_ylabel("Times chance (accuracy × N)")
    b.set_title("Relative to chance")
    handles = [Line2D([], [], color=c, lw=2.4, marker="o", label=f"{lbl}, Lee2019 subsets")
               for _, _, lbl, c in N_SERIES] + [
        Line2D([], [], color=C["muted"], marker="D", ls="", mfc="white", mew=2, ms=8,
               label="Same model on BCI IV 2a (9 users, experiment B)"),
        Line2D([], [], color=C["muted"], lw=8, alpha=0.3, label="95% participant-bootstrap interval"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.045),
               columnspacing=3.0, fontsize=NOTE_SIZE + 0.5)
    save(fig, "fig7_lee2019_users")


def main():
    fig_pitfall1()
    fig_frontier()
    fig_commands()
    fig_enrolment()
    fig_timeline()
    fig_lee2019()
    fig_crowd()
    for p in sorted(OUT.glob("*")):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
