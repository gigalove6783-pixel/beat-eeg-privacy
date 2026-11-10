"""Summary figures for slides, README and the technical report.

Builds five figures from committed result files only and writes PNG (200 dpi)
and SVG to ``results/figures/summary/``:

    python scripts/make_summary_figures.py

Inputs: results/exp_b, exp_c, exp_d, exp_e, exp_g/bnci tables and the seal
records in docs/decision_log.md. ``results/exp_l/`` is deliberately not read;
``panel_lee2019`` is a stub for a later panel.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
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
    seal3_date, m3 = find(r"\*\*Seal 3\*\*: commit `([0-9a-f]{7})")
    if not (m1 and m2 and m_ab and g_date):
        raise ValueError("decision_log.md: expected seal/result entries not found")
    return dict(
        seal1=(m1.group(1), seal1_date, m1.group(2)),
        seal2=(m2.group(1), seal2_date, m2.group(2)),
        ab=(m_ab.group(1), ab_date),
        g=g_date,
        seal3=(m3.group(1), seal3_date) if m3 else None,
    )


def fig_timeline():
    s = load_seals()
    fig = new_figure(
        "Confirmatory claims are sealed before their evaluation data are opened",
        "Order of seals and analyses in docs/decision_log.md; exploratory work reuses data "
        "already seen and makes no confirmatory claim.",
        "CONFIRMATORY + EXPLORATORY",
        "docs/decision_log.md (commits, dates, test counts); status per docs/report.md §4.7, §5",
    )
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.78])
    ax.set_xlim(-0.6, 7.0)
    ax.set_ylim(-2.0, 1.05)
    ax.axis("off")
    ax.annotate("", xy=(6.98, 0), xytext=(-0.55, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6), zorder=1)

    seal3 = s["seal3"]
    seals = [
        (0, "E seal 1", f"{s['seal1'][0]} · {s['seal1'][1]}\nplan v1 · {s['seal1'][2]} tests passed",
         True),
        (2, "E seal 2", f"{s['seal2'][0]} · {s['seal2'][1]}\nplan v2 · {s['seal2'][2]} tests passed",
         True),
        (5.3, "Seal 3",
         f"{seal3[0]} · {seal3[1]}\nplan v3" if seal3 else "not yet recorded\nplan v3 (claims C1–C4)",
         bool(seal3)),
    ]
    for x, name, detail, done in seals:
        colour = "black" if done else C["muted"]
        ax.scatter(x, 0, marker="D", s=230, color=colour if done else "white",
                   edgecolor=colour, lw=2, zorder=3)
        ax.text(x, 0.62, name, ha="center", va="bottom", size=LABEL_SIZE + 1, weight="bold",
                color=colour)
        ax.text(x, 0.56, detail, ha="center", va="top", size=NOTE_SIZE + 0.5, color=colour,
                linespacing=1.4)

    status_colour = {"Confirmatory": C["head"], "Exploratory": C["grl"],
                     "Confirmatory\n(pending)": C["pending"]}
    analyses = [
        (1, "A · B · C", "Confirmatory",
         f"A, B final at {s['ab'][0]};\nH1, H2 not supported;\ncommands leak\nabove chance"),
        (3, "D · E", "Exploratory",
         "reuse session E;\nno defence passes;\nhead–attacker\ngap recurs"),
        (4.2, "G · published\ndefence", "Exploratory",
         "no seal; paper's\noperating point;\nre-identified with\n1–5 labels"),
        (6.4, "L · Lee2019", "Confirmatory\n(pending)",
         "54 users; session 2\nunopened; claims\nC1–C4 by Holm,\nα = 0.05"),
    ]
    box_w, box_h, top = 1.05, 1.48, -0.3
    for x, name, status, detail in analyses:
        colour = status_colour[status]
        ax.plot([x, x], [0, top], color=colour, lw=1.5, zorder=1)
        ax.scatter(x, 0, s=70, color=colour, zorder=3)
        ax.add_patch(FancyBboxPatch((x - box_w / 2, top - box_h), box_w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec=colour, lw=2.2, zorder=2,
                                    ls="--" if "pending" in status else "-"))
        ax.text(x, top - 0.1, name, ha="center", va="top", size=LABEL_SIZE,
                weight="bold", zorder=3, linespacing=1.15)
        ax.text(x, top - 0.55, status.upper(), ha="center", va="top", size=NOTE_SIZE - 0.5,
                weight="bold", color=colour if "pending" not in status else C["muted"],
                zorder=3)
        ax.text(x, top - 0.86, detail, ha="center", va="top", size=NOTE_SIZE,
                color="#333333", zorder=3, linespacing=1.3)
    save(fig, "fig5_preregistration_timeline")


# ----------------------------------------------------------------------------
# Later: Lee2019 / OpenBMI (experiment L). Not called; results/exp_l/ is not read.
# ----------------------------------------------------------------------------
def panel_lee2019(ax):
    """Placeholder for the Lee2019 confirmatory panel (claims C1-C4).

    Intended content once seal 3 and the L final run exist: per-claim observed
    effect against its pre-registered threshold, Holm-adjusted decision, chance
    1/54. Sources will be results/exp_l/decision.json and identity_summary.csv.
    """
    raise NotImplementedError("Lee2019 results are pending (seal 3 not recorded).")


def main():
    fig_pitfall1()
    fig_frontier()
    fig_commands()
    fig_enrolment()
    fig_timeline()
    for p in sorted(OUT.glob("*")):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
