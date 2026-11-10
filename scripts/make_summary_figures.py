"""Summary figures for slides, README and the technical report.

Builds nine figures from committed result files only and writes PNG (200 dpi) and SVG,
in English to ``results/figures/summary/`` and in Korean to ``results/figures/summary_ko/``:

    python scripts/make_summary_figures.py [--lang en|ko|all]    (default: all)

Inputs: results/exp_b, exp_c, exp_d, exp_e, exp_g/bnci, exp_k and exp_l tables and the
seal records in docs/decision_log.md. The five-seed co-trained head score of
experiment L is recomputed from results/exp_l/head_predictions.csv.gz with the
project's own ``beat.exp_l.recalls`` and participant bootstrap ``interval``.
Figures 8 and 9 (experiment K, exploratory) plot rows of results/exp_k/identity_curves.csv
and results/exp_k/frontier.csv as stored.

The Korean set shows the same data and numbers in larger type for slides. It uses the
Malgun Gothic font (Arial supplies the glyphs Malgun Gothic lacks, such as U+2212); its SVGs
store glyphs as paths, so they render without the font installed.
"""
from __future__ import annotations

import argparse
import io
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
OUT_DIRS = {"en": RES / "figures" / "summary", "ko": RES / "figures" / "summary_ko"}

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
# Type sizes in points; the Korean set is larger so that it stays legible on a slide.
SIZES = {
    "en": dict(title=18, subtitle=13, label=13, tick=11.5, note=10.5),
    "ko": dict(title=24, subtitle=17, label=17, tick=15.5, note=15.5),
}
# Korean lists the families in font.family itself: matplotlib falls back glyph by glyph only
# across that list (a generic "sans-serif" resolves to one font), and Malgun Gothic lacks U+2212.
FONT_FAMILY = {"en": "sans-serif", "ko": ["Malgun Gothic", "Arial", "DejaVu Sans"]}
# Korean SVGs store glyphs as paths: SVG viewers need not have Malgun Gothic.
SVG_FONTTYPE = {"en": "none", "ko": "path"}
KO_TAGS = {
    "CONFIRMATORY": "확증",
    "EXPLORATORY": "탐색적",
    "DESCRIPTIVE": "기술 통계",
    "CONFIRMATORY + EXPLORATORY": "확증 + 탐색적",
    "B CONFIRMATORY · D, E EXPLORATORY": "B 확증 · D, E 탐색적",
    "B GRL CONFIRMATORY · D EXPLORATORY": "B GRL 확증 · D 탐색적",
}
LANG = OUT = TITLE_SIZE = SUBTITLE_SIZE = LABEL_SIZE = TICK_SIZE = NOTE_SIZE = None


def set_language(lang):
    """Select labels, type sizes, fonts and output folder for ``lang`` ("en" or "ko")."""
    global LANG, OUT, TITLE_SIZE, SUBTITLE_SIZE, LABEL_SIZE, TICK_SIZE, NOTE_SIZE
    LANG, OUT = lang, OUT_DIRS[lang]
    s = SIZES[lang]
    TITLE_SIZE, SUBTITLE_SIZE, LABEL_SIZE = s["title"], s["subtitle"], s["label"]
    TICK_SIZE, NOTE_SIZE = s["tick"], s["note"]
    plt.rcParams.update({
        "font.family": FONT_FAMILY[lang],
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
        "svg.fonttype": SVG_FONTTYPE[lang],
        "svg.hashsalt": "beat-summary",
    })


def tr(en, ko):
    """The English or Korean variant (label or layout value) for the current language."""
    return ko if LANG == "ko" else en


set_language("en")


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def new_figure(title, subtitle, tag, source, **subplot_kw):
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.02, 0.975, title, size=TITLE_SIZE, weight="bold", va="top")
    fig.text(0.02, tr(0.915, 0.9), subtitle, size=SUBTITLE_SIZE, color=C["muted"], va="top")
    tag_colour = {"CONFIRMATORY": C["head"], "EXPLORATORY": C["grl"]}.get(tag, C["muted"])
    fig.text(0.985, 0.018, tr(tag, KO_TAGS[tag]), size=tr(NOTE_SIZE, 14), weight="bold",
             color="white", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", fc=tag_colour, ec="none"))
    fig.text(0.02, 0.018, tr("Source: ", "자료: ") + source, size=tr(NOTE_SIZE - 1, 12.5),
             color=C["muted"], va="bottom")
    return fig


def panel_tag(ax, text):
    """Status tag on the panel's title line, right-aligned above the axes."""
    colour = {"CONFIRMATORY": C["head"], "EXPLORATORY": C["grl"]}[text]
    ax.text(1.0, 1.035, tr(text, KO_TAGS[text]), transform=ax.transAxes, ha="right",
            va="bottom", size=NOTE_SIZE - 1.5, weight="bold", color=colour,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour, lw=1))


def chance_handle():
    return Line2D([], [], color=C["chance"], ls=":", lw=1.6,
                  label=tr("Chance (1/9)", "우연 수준 (1/9)"))


def chance_line(ax, label=True, x=0.01):
    ax.axhline(CHANCE, color=C["chance"], ls=":", lw=1.4, zorder=1)
    if label:
        ax.text(x, CHANCE + 0.012, tr("chance (1/9)", "우연 수준 (1/9)"),
                transform=ax.get_yaxis_transform(), size=NOTE_SIZE, color=C["chance"],
                va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, metadata={"Software": None})
    # Through a buffer: matplotlib writes SVG in text mode, which gives CRLF on Windows (Git: LF).
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    (OUT / f"{name}.svg").write_text(svg.getvalue(), encoding="utf-8", newline="\n")
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

    common, recentred = tr("Common features", "공통 특징"), tr("Recentred features", "재중심화 특징")
    return [
        dict(title=common, tag="CONFIRMATORY", series=[b_series("P3")]),
        dict(title=common, tag="EXPLORATORY",
             series=[d_series("MaxEnt", "maxent"), d_series("Ensemble", "ensemble")]),
        dict(title=tr("No alignment", "정렬 없음"), tag="EXPLORATORY",
             series=[e_series("none")]),
        dict(title=recentred, tag="CONFIRMATORY", series=[b_series("P3-RC")]),
        dict(title=recentred, tag="EXPLORATORY",
             series=[d_series("MaxEnt-RC", "maxent"), d_series("Ensemble-RC", "ensemble")]),
        dict(title=tr("Euclidean alignment", "유클리드 정렬"), tag="EXPLORATORY",
             series=[e_series("ea")]),
    ]


PITFALL1_COLUMNS = [("B · Gradient reversal", "B · gradient reversal"),
                    ("D · MaxEnt and ensemble-refresh", "D · MaxEnt, ensemble-refresh"),
                    ("E · EEGNet with co-trained identifier", "E · EEGNet + 함께 학습한 식별기")]
VARIANT_MARKER = {"grl": "o", "maxent": "^", "ensemble": "s"}


def fig_pitfall1():
    panels = load_pitfall1()
    b = pd.read_csv(RES / "exp_b" / "adversary_comparison.csv")
    at1 = b[(b.condition == "P3") & (b.target == 1.0)].iloc[0]
    fig = new_figure(
        tr("Pitfall 1: the co-trained identity head says “private”; independent attackers disagree",
           "함정 1: 함께 학습한 식별기는 ‘안전’, 독립 공격자는 다시 맞힌다"),
        tr(f"Same frozen outputs, two judges. B at λ = 1: own head {at1.adversary_main_ba:.2f} "
           f"(below chance) vs retrained attacker {at1.independent_ba:.2f}.",
           f"같은 고정 출력, 두 심판. B, λ = 1: 함께 학습한 식별기 {at1.adversary_main_ba:.2f} "
           f"(우연 미만) 대 재학습 공격자 {at1.independent_ba:.2f}"),
        "B CONFIRMATORY · D, E EXPLORATORY",
        "results/exp_b/adversary_comparison.csv, results/exp_d/tradeoff.csv, "
        "results/exp_e/lambda_summary.csv",
    )
    axes = fig.subplots(2, 3, sharey=True)
    fig.subplots_adjust(left=tr(0.075, 0.08), right=0.985, top=tr(0.765, 0.67),
                        bottom=tr(0.215, 0.3), wspace=0.08, hspace=tr(0.45, 0.75))
    for ax, header in zip(axes[0], PITFALL1_COLUMNS):
        ax.text(0, tr(1.2, 1.45), tr(*header), transform=ax.transAxes,
                size=tr(LABEL_SIZE + 1.5, LABEL_SIZE), weight="bold", va="bottom")
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
        if LANG == "ko":
            ax.set_yticks([0, 0.5, 1])
        ax.set_title(spec["title"], pad=7, weight="normal")
        chance_line(ax, label=False)
        panel_tag(ax, spec["tag"])
    for ax in axes[1]:
        ax.set_xlabel(tr("Adversarial strength λ", "적대 강도 λ"))
    if LANG == "ko":
        # One label for both rows: a per-row label does not fit the shorter Korean rows.
        fig.supylabel("신원 균형 정확도", x=0.012, y=0.485, size=LABEL_SIZE)
    else:
        for ax in axes[:, 0]:
            ax.set_ylabel("Identity balanced accuracy")

    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=C["head"], mew=1.8, ms=8,
               label=tr("Co-trained identity head (the model's own judge)",
                        "함께 학습한 식별기 (모델 자신의 심판)")),
        Line2D([], [], marker="o", ls="", color=C["attacker"], ms=8,
               label=tr("Strongest independently retrained attacker",
                        "가장 강한 독립 재학습 공격자")),
        Line2D([], [], color="#C8C8C8", lw=4,
               label=tr("Gap = exposure the head misses", "간격 = 식별기가 놓친 노출")),
        Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=8,
               label=tr("Gradient reversal (B, E)", "gradient reversal (B, E)")),
        Line2D([], [], marker="^", ls="", mfc="none", mec="black", ms=8, label="MaxEnt (D)"),
        Line2D([], [], marker="s", ls="", mfc="none", mec="black", ms=8,
               label=tr("Ensemble-refresh (D)", "ensemble-refresh (D)")),
        chance_handle(),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=tr(4, 3),
               bbox_to_anchor=(0.5, tr(0.05, 0.045)), handletextpad=0.4,
               columnspacing=tr(1.6, 1.2))
    save(fig, "fig1_pitfall1_head_vs_attacker")


# ----------------------------------------------------------------------------
# Figure 2: utility-identity frontier
# ----------------------------------------------------------------------------
def load_frontier():
    b = pd.read_csv(RES / "exp_b" / "tradeoff.csv")
    d = pd.read_csv(RES / "exp_d" / "tradeoff.csv")
    out = {}
    for fam, b_cond, label in (("common", "P3", tr("Common features", "공통 특징")),
                               ("rc", "P3-RC", tr("Recentred features (per-user offset removed)",
                                                  "재중심화 특징 (사용자별 오프셋 제거)"))):
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
# Korean labels are taller: d = 2 sits above-left so its box clears the d = 1 point and line.
D_LABEL_OFFSET_KO = {**D_LABEL_OFFSET, ("rc", 1): (9, -4, "left"), ("rc", 2): (-9, 8, "right")}


def fig_frontier():
    data = load_frontier()
    d1 = {f: data[f]["task"].iloc[0] for f in data}
    fig = new_figure(
        tr("No adversarial defence beats a narrow task-only bottleneck",
           "어떤 적대적 방어도 좁은 과제 전용 병목을 넘지 못했다"),
        tr(f"Lowest identity at full utility: a 1-d bottleneck ({d1['common'].identity_ba:.2f} "
           f"common, {d1['rc'].identity_ba:.2f} recentred) — still about twice chance.",
           f"유용성을 유지한 가장 낮은 식별: 1차원 병목 (공통 {d1['common'].identity_ba:.2f}, "
           f"재중심화 {d1['rc'].identity_ba:.2f}) — 여전히 우연의 약 2배"),
        "B GRL CONFIRMATORY · D EXPLORATORY",
        tr("results/exp_d/tradeoff.csv (task-only, MaxEnt, ensemble, P0), "
           "results/exp_b/tradeoff.csv (GRL)",
           "results/exp_d/tradeoff.csv (과제 전용, MaxEnt, ensemble, P0), "
           "results/exp_b/tradeoff.csv (GRL)"),
    )
    axes = fig.subplots(1, 2, sharey=True, sharex=True)
    fig.subplots_adjust(left=tr(0.075, 0.085), right=0.985, top=tr(0.81, 0.77),
                        bottom=tr(0.235, 0.27), wspace=0.07)
    if LANG == "ko":  # keeps the first x tick label off the "0.0" y tick label
        axes[0].tick_params(axis="x", pad=7)
    for ax, fam in zip(axes, ("common", "rc")):
        spec = data[fam]
        task = spec["task"]
        ax.plot(task.task_ba, task.identity_ba, "-o", color=C["task"], lw=2.2, ms=7, zorder=4)
        for _, r in task.iterrows():
            # Labels sit left of the line; adversarial points cluster to its right.
            dx, dy, ha = tr(D_LABEL_OFFSET, D_LABEL_OFFSET_KO).get((fam, int(r.d)),
                                                                   (-9, 0, "right"))
            ax.annotate(f"d = {int(r.d)}", (r.task_ba, r.identity_ba), xytext=(dx, dy),
                        textcoords="offset points", size=tr(NOTE_SIZE, 14), color=C["task"],
                        weight="bold", ha=ha, va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        raw = spec["raw"]
        ax.scatter(raw.task_ba, raw.identity_ba, marker="*", s=260, color=C["raw"], zorder=5)
        ax.annotate(tr(f"{raw.condition}: full 253-d vector", f"{raw.condition}: 253차원 전체 벡터"),
                    (raw.task_ba, raw.identity_ba), xytext=(10, -4), textcoords="offset points",
                    size=NOTE_SIZE)
        for key, df in spec["adversarial"].items():
            ax.scatter(df.task_ba, df.identity_ba, marker=VARIANT_MARKER[key], s=58,
                       color=C[key], edgecolor="white", lw=0.6, zorder=3, alpha=0.95)
            sel = df[df.selected.astype(bool)]
            ax.scatter(sel.task_ba, sel.identity_ba, marker="o", s=230, facecolor="none",
                       edgecolor="black", lw=1.6, zorder=6)
        chance_line(ax)
        ax.set_title(spec["label"], pad=6)
        ax.set_xlabel(tr("Task balanced accuracy (left vs right; higher = more useful)",
                         "과제 균형 정확도 (왼쪽 대 오른쪽, 높을수록 유용)"))
        ax.text(0.985, 0.04, tr("better: right and down →", "더 좋음: 오른쪽 아래 →"),
                transform=ax.transAxes, ha="right", size=NOTE_SIZE, color=C["muted"],
                style=tr("italic", "normal"))
    # Korean y labels are single lines: matplotlib 3.11 ignores line spacing in rotated
    # multi-line text, and the taller Hangul lines then overlap.
    axes[0].set_ylabel(tr("Identity balanced accuracy\n(strongest independent attacker)",
                          "신원 균형 정확도 (최강 독립 공격자)"), **tr({}, {"size": 15}))
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xlim(0.64, 0.735)
    handles = [
        Line2D([], [], marker="*", ls="", color=C["raw"], ms=13,
               label=tr("No defence (P0 / P0-RC)", "방어 없음 (P0 / P0-RC)")),
        Line2D([], [], marker="o", color=C["task"], lw=2.2, ms=7,
               label=tr("Task-only bottleneck, width d", "과제 전용 병목, 너비 d")),
        Line2D([], [], marker="o", ls="", color=C["grl"], ms=8,
               label=tr("Gradient reversal (B), all λ", "gradient reversal (B), 모든 λ")),
        Line2D([], [], marker="^", ls="", color=C["maxent"], ms=8,
               label=tr("MaxEnt (D), all λ", "MaxEnt (D), 모든 λ")),
        Line2D([], [], marker="s", ls="", color=C["ensemble"], ms=8,
               label=tr("Ensemble-refresh (D), all λ", "ensemble-refresh (D), 모든 λ")),
        Line2D([], [], marker="o", ls="", mfc="none", mec="black", mew=1.6, ms=13,
               label=tr("λ selected on session T", "세션 T에서 선택한 λ")),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055),
               columnspacing=tr(2.0, 1.2))
    save(fig, "fig2_utility_identity_frontier")


# ----------------------------------------------------------------------------
# Figure 3: command accumulation
# ----------------------------------------------------------------------------
# Ordered so the three-column legend reads: common | recentred | references.
C_MODELS = [
    ("P0", ("No defence (P0)", "방어 없음 (P0)"), C["raw"], "-"),
    ("P2", ("Bottleneck d = 8 (P2)", "병목 d = 8 (P2)"), C["task"], "-"),
    ("P3", ("GRL, T-selected λ (P3)", "GRL, 세션 T에서 선택한 λ (P3)"), C["grl"], "-"),
    ("P0-RC", ("No defence, recentred (P0-RC)", "방어 없음, 재중심화 (P0-RC)"), C["raw"], "--"),
    ("P2-RC", ("Bottleneck, recentred (P2-RC)", "병목, 재중심화 (P2-RC)"), C["task"], "--"),
    ("P3-RC", ("GRL, recentred (P3-RC)", "GRL, 재중심화 (P3-RC)"), C["grl"], "--"),
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
        tr("Even bare left / right / hold commands identify users as they accumulate",
           "왼쪽·오른쪽·정지 명령만 보내도, 쌓이면 사용자가 드러난다"),
        tr(f"With 5 consecutive commands, {data['n_by5']} of {data['n_settings']} model × "
           "hold-threshold settings exceed twice chance (point estimates).",
           f"연속 명령 5개에서 모델 × 정지 임계값 설정 {data['n_settings']}개 중 "
           f"{data['n_by5']}개가 우연의 2배를 넘는다 (점추정)"),
        "CONFIRMATORY",
        "results/exp_c/observation_strongest.csv, results/exp_c/observation_threshold.csv",
    )
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=tr(0.075, 0.085), right=0.985, top=tr(0.81, 0.77),
                        bottom=tr(0.27, 0.32), wspace=tr(0.17, 0.2))
    ks = [1, 5, 10, 20]
    for ax, key, title, ylim in (
        (axes[0], "feature", tr("Transmitting features", "특징을 전송할 때"), (0, 1.05)),
        (axes[1], "command", tr("Transmitting only commands (hold threshold τ = 0.5)",
                                "명령만 전송할 때 (정지 임계값 τ = 0.5)"), (0, 0.6)),
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
        ax.set_xlabel(tr("Consecutive outputs observed from one user (k)",
                         "한 사용자에게서 관찰한 연속 출력 수 (k)"))
    axes[0].set_ylabel(tr("Identity balanced accuracy\n(strongest attacker and scenario)",
                          "신원 균형 정확도 (최강 공격자·시나리오)"), **tr({}, {"size": 14.5}))
    handles = [Line2D([], [], ls=ls, marker="o", color=c, lw=2.2, ms=6, label=tr(*lbl))
               for _, lbl, c, ls in C_MODELS]
    handles += [
        Line2D([], [], color="#D9D9D9", lw=9,
               label=tr("Commands, range over all models and τ\n(max over within-session and cross-day)",
                        "명령: 모든 모델·τ에 걸친 범위\n(같은 세션·다른 날 중 최댓값)")),
        Line2D([], [], color=C["chance"], ls="--", lw=1.4,
               label=tr("Twice chance (2/9)", "우연의 2배 (2/9)")),
        chance_handle(),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04),
               columnspacing=tr(3.0, 1.5), fontsize=tr(NOTE_SIZE + 0.5, 14.5),
               labelspacing=0.35)
    save(fig, "fig3_command_accumulation")


# ----------------------------------------------------------------------------
# Figure 4: Pitfall 2, experiment G
# ----------------------------------------------------------------------------
G_ATTACKERS = [("beat_logistic", ("logistic regression", "로지스틱 회귀"), "-"),
               ("paper_ts_svm", ("paper's tangent-space SVM", "논문의 접공간 SVM"), "--")]
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
        tr("Pitfall 2: a defence tested only against reference-session attackers",
           "함정 2: 기준 세션 공격자로만 시험한 방어"),
        tr(f"The published defence reproduces (11.1%), but one labelled deployment trial per user "
           f"restores {n1:.0%} identification, five restore {n5:.0%}.",
           f"공개 방어는 재현되지만(11.1%), 사용자당 배포 기간 라벨 1개로 {n1:.0%}, "
           f"5개로 {n5:.0%}가 다시 식별된다"),
        "EXPLORATORY",
        "results/exp_g/bnci/enrolment_averaged.csv, identity_averaged.csv, "
        "paper_table1_comparison.csv",
    )
    ax = fig.subplots()
    fig.subplots_adjust(left=tr(0.085, 0.1), right=tr(0.70, 0.63), top=tr(0.83, 0.79),
                        bottom=tr(0.22, 0.25))
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
    ax.annotate(tr(f"Paper's protocol: {paper.paper_accuracy_percent:.1f}% reported,\n"
                   f"{paper.observed_accuracy_percent:.1f}% reproduced here (session 1 → 2)",
                   f"논문 프로토콜: 보고 {paper.paper_accuracy_percent:.1f}%,\n"
                   f"여기서 재현 {paper.observed_accuracy_percent:.1f}% (세션 1 → 2)"),
                (star_x, star_y), xytext=(tr(0.75, 0.7), 0.36), textcoords="data",
                size=NOTE_SIZE + 0.5, va="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1, shrinkB=9))
    ax.axvspan(-0.5, 0.5, color="#F2F2F2", zorder=0)
    chance_line(ax, x=tr(0.83, 0.74))
    ax.set_xticks([0] + xs, [tr("0\n(reference\nsession only)", "0\n(기준\n세션만)")]
                  + G_N[:-1] + [tr("all", "전체")])
    ax.set_xlim(-0.5, len(G_N) + 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(tr("Labelled deployment-period trials per user available to the attacker",
                     "공격자가 가진 사용자당 배포 기간 라벨 시행 수"))
    ax.set_ylabel(tr("Users correctly identified\n(balanced accuracy, single trial)",
                     "사용자 식별 균형 정확도 (단일 시행)"))

    handles = [
        Line2D([], [], color=C["raw"], lw=2.4, marker="o",
               label=tr("Undefended data", "방어 없는 데이터")),
        Line2D([], [], color=C["head"], lw=2.4, marker="o",
               label=tr("Defended data\n(transform fitted on reference session)",
                        "방어된 데이터\n(기준 세션에서 학습한 변환)")),
    ] + [Line2D([], [], color="black", ls=ls, lw=2,
                label=tr(f"Attacker: {lbl[0]}", f"공격자: {lbl[1]}"))
         for _, lbl, ls in G_ATTACKERS] + [
        Line2D([], [], marker="*", ls="", ms=16, mfc=C["grl"], mec="black",
               label=tr("Paper's reported result", "논문 보고 결과")),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(tr(0.715, 0.645), 0.53),
               labelspacing=1.1)
    fig.text(tr(0.715, 0.645), tr(0.20, 0.14),
             tr("BCI IV 2a, 9 users; both session\ndirections averaged; "
                "RBF-SVM and MLP\nattackers need ≥ 3 labels (not shown).",
                "BCI IV 2a, 9명, 두 세션 방향 평균\nRBF-SVM·MLP 공격자는 라벨이\n"
                "3개 이상 필요 (미표시)"),
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
        tr("Confirmatory claims are sealed before their evaluation data are opened",
           "확증 주장은 평가 데이터를 열기 전에 봉인한다"),
        tr("Order of seals and analyses in docs/decision_log.md; exploratory work reuses data "
           "already seen and makes no confirmatory claim.",
           "docs/decision_log.md의 봉인·분석 순서. 탐색적 분석은 이미 본 데이터를 재사용하며 "
           "확증 주장을 하지 않는다"),
        "CONFIRMATORY + EXPLORATORY",
        tr("docs/decision_log.md (commits, dates, test counts, run attempts); "
           "L decisions from results/exp_l/decision.json",
           "docs/decision_log.md (커밋, 날짜, 테스트 수, 실행 시도); "
           "L 판정: results/exp_l/decision.json"),
    )
    ax = fig.add_axes([0.02, 0.08, 0.96, tr(0.78, 0.76)])
    ax.set_xlim(-0.6, 7.0)
    ax.set_ylim(tr(-2.0, -2.4), tr(1.05, 1.15))
    ax.axis("off")
    ax.annotate("", xy=(6.98, 0), xytext=(-0.55, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6), zorder=1)

    s1, s2, s3 = s["seal1"], s["seal2"], s["seal3"]
    seals = tr([
        (0, "E seal 1", f"{s1[0]} · {s1[1]}\nplan v1 · {s1[2]} tests passed"),
        (2, "E seal 2", f"{s2[0]} · {s2[1]}\nplan v2 · {s2[2]} tests passed"),
        (5.3, "Seal 3", f"{s3[0]} · {s3[1]}\nplan v3 · {s3[2]} tests passed"),
    ], [
        (0, "E 봉인 1", f"{s1[0]} · {s1[1]}\n계획 v1\n테스트 {s1[2]}개 통과"),
        (2, "E 봉인 2", f"{s2[0]} · {s2[1]}\n계획 v2\n테스트 {s2[2]}개 통과"),
        (5.3, "봉인 3", f"{s3[0]} · {s3[1]}\n계획 v3\n테스트 {s3[2]}개 통과"),
    ])
    for x, name, detail in seals:
        ax.scatter(x, 0, marker="D", s=230, color="black", lw=2, zorder=3)
        ax.text(x, tr(0.62, 0.74), name, ha="center", va="bottom", size=LABEL_SIZE + 1,
                weight="bold")
        ax.text(x, tr(0.56, 0.68), detail, ha="center", va="top", size=tr(NOTE_SIZE + 0.5, 15),
                linespacing=1.4)

    status_colour = {"Confirmatory": C["head"], "Exploratory": C["grl"]}
    status_ko = {"Confirmatory": "확증", "Exploratory": "탐색적"}
    analyses = tr([
        (1, "A · B · C", "Confirmatory",
         f"A, B final at {s['ab'][0]};\nH1, H2 not supported;\ncommands leak\nabove chance"),
        (3, "D · E", "Exploratory",
         "reuse session E;\nno defence passes;\nhead–attacker\ngap recurs"),
        (4.2, "G · published\ndefence", "Exploratory",
         "no seal; paper's\noperating point;\nre-identified with\n1–5 labels"),
        (6.4, "L · Lee2019", "Confirmatory",
         f"54 users; {', '.join(supported)}\nsupported; {', '.join(rejected)} not\n"
         "supported (Holm)"),
    ], [
        (1, "A · B · C", "Confirmatory",
         f"A, B 최종 커밋\n{s['ab'][0]}\nH1, H2 지지 안 됨\n명령 누출은\n우연 이상"),
        (3, "D · E", "Exploratory", "세션 E 재사용\n통과한 방어 없음\n식별기–공격자\n간격 반복"),
        (4.2, "G · 공개\n방어", "Exploratory", "봉인 없음\n논문 운영점\n라벨 1–5개로\n재식별"),
        (6.4, "L · Lee2019", "Confirmatory",
         f"54명\n{', '.join(supported)} 지지\n{', '.join(rejected)} 지지 안 됨\n(Holm 보정)"),
    ])
    box_w, box_h, top = tr(1.05, 1.15), tr(1.56, 2.0), -0.3
    for x, name, status, detail in analyses:
        colour = status_colour[status]
        ax.plot([x, x], [0, top], color=colour, lw=1.5, zorder=1)
        ax.scatter(x, 0, s=70, color=colour, zorder=3)
        ax.add_patch(FancyBboxPatch((x - box_w / 2, top - box_h), box_w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec=colour, lw=2.2, zorder=2))
        ax.text(x, top - 0.1, name, ha="center", va="top", size=LABEL_SIZE,
                weight="bold", zorder=3, linespacing=1.15)
        ax.text(x, top - 0.55, tr(status.upper(), status_ko[status]), ha="center", va="top",
                size=NOTE_SIZE - 0.5, weight="bold", color=colour, zorder=3)
        ax.text(x, top - tr(0.86, 0.78), detail, ha="center", va="top",
                size=tr(NOTE_SIZE, 14), color="#333333", zorder=3, linespacing=1.3)
    if s["l_rerun"]:
        ax.text(6.4, top - box_h + 0.07,
                tr("run 1 stopped by sealed\nmemory guard; unchanged\nre-run",
                   "1차 실행은 봉인된\n메모리 가드로 중단,\n변경 없이 재실행"),
                ha="center", va="bottom", size=tr(NOTE_SIZE - 1.5, 13), color=C["muted"],
                style=tr("italic", "normal"), zorder=3, linespacing=1.2)
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
ATTACKER_NAME_KO = {"logistic": "로지스틱 회귀", "beat_logistic": "로지스틱 회귀",
                    "rbf_svm": "RBF-SVM", "beat_rbf_svm": "RBF-SVM", "mlp": "MLP",
                    "beat_mlp": "MLP", "paper_ts_svm": "논문의 접공간 SVM"}


def attacker_name(key):
    return tr(ATTACKER_NAME[key], ATTACKER_NAME_KO[key])


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
    verdict = tr("SUPPORTED", "지지됨") if supported else tr("NOT SUPPORTED", "지지 안 됨")
    text = verdict + tr(f"  ·  Holm p = {fmt_p(claim['holm_p'])}",  # Korean: room for titles
                        f" · Holm p = {fmt_p(claim['holm_p']).replace(' × ', '×')}")
    ax.text(1.0, 1.04, text, transform=ax.transAxes, ha="right", va="bottom",
            size=tr(NOTE_SIZE - 0.5, 14), weight="bold", color="white" if supported else "black",
            bbox=dict(boxstyle="round,pad=0.3", fc="#222222" if supported else "white",
                      ec="#222222", lw=1.2))


def interval_rows(ax, rows, xmax, gap_below=None):
    """Rows top to bottom: (label, value, low, high, colour[, marker_x]). Label sits above its
    interval. Korean only: offsets are in points, so the larger type clears its own value at
    any panel height; the value also clears an extra marker drawn at marker_x on its row, and
    gap_below adds space below that row index. Returns the row heights."""
    n = len(rows)
    extra = 0.7 if gap_below is not None else 0
    ys = []
    for i, (label, v, lo, hi, colour, *mark) in enumerate(rows):
        y = n - 1 - i + (extra if gap_below is not None and i <= gap_below else 0)
        ys.append(y)
        ax.plot([lo, hi], [y, y], color=colour, lw=3.2, alpha=0.45, solid_capstyle="butt",
                zorder=3)
        ax.scatter(v, y, s=60, color=colour, zorder=4, clip_on=False)
        right = hi + 0.012 * xmax < 0.86 * xmax
        if LANG == "ko":
            dx = 11 if mark else 5
            ax.annotate(f"{v:.3f}", (max([hi, *mark]) if right else lo, y),
                        xytext=(dx if right else -dx, 0), textcoords="offset points",
                        ha="left" if right else "right", va="center", size=NOTE_SIZE,
                        weight="bold", color=colour, zorder=4)
            ax.annotate(label, (0.004 * xmax, y), xytext=(0, 0.5 * NOTE_SIZE + 3),
                        textcoords="offset points", ha="left", va="bottom", size=13.5,
                        color="#333333", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
            continue
        ax.text(hi + 0.012 * xmax if right else lo - 0.012 * xmax, y, f"{v:.3f}",
                ha="left" if right else "right", va="center", size=NOTE_SIZE, weight="bold",
                color=colour, zorder=4)
        ax.text(0.004 * xmax, y + 0.4, label, ha="left", va="bottom", size=NOTE_SIZE,
                color="#333333", zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_ylim(*((-0.35, n + 0.3 + extra) if extra else (-0.55, n - 0.05)))
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.axvline(L_CHANCE, color=C["chance"], ls=":", lw=1.6, zorder=1)
    return ys


def threshold_line(ax, x):
    ax.axvline(x, color="black", ls="-.", lw=1.3, zorder=2)


def pp(x):
    return f"{x * 100:+.1f}".replace("-", "−") + " pp"


def fig_lee2019():
    d = load_lee2019()
    dec = d["decision"]
    ok = sum(v["decision"] == "supported" for v in dec.values())
    fig = new_figure(
        tr(f"Confirmatory replication on 54 users: {ok} of {len(dec)} pre-registered claims hold",
           f"54명 확증 재현: 사전등록 주장 {len(dec)}개 중 {ok}개 지지"),
        tr("Lee2019/OpenBMI, five seeds, Holm at α = 0.05. Both pitfalls and the bottleneck result "
           "replicate; command accumulation does not pass correction.",
           "Lee2019/OpenBMI, seed 5개, Holm α = 0.05 · 두 함정과 병목 결과는 재현, "
           "명령 누적은 보정 후 미통과"),
        "CONFIRMATORY",
        "results/exp_l/decision.json, identity_summary.csv, task_summary.csv, "
        "head_predictions.csv.gz, reproduction.json",
    )
    axes = fig.subplots(2, 2, gridspec_kw=dict(height_ratios=tr([1, 1.3], [1.35, 1.6])))
    fig.subplots_adjust(left=tr(0.035, 0.03), right=0.985, top=tr(0.80, 0.76),
                        bottom=tr(0.175, 0.225), wspace=tr(0.10, 0.08), hspace=tr(0.5, 0.5))
    (a1, a2), (a3, a4) = axes

    # C1: co-trained head vs strongest independent attacker (P3, λ = 1).
    h, att = d["head"], d["c1_att"]
    interval_rows(a1, [
        (tr("Co-trained identity head (gradient reversal, λ = 1)",
            "함께 학습한 식별기 (gradient reversal, λ = 1)"), h["ba"], h["ci_low"],
         h["ci_high"], C["head"]),
        (tr(f"Strongest independent attacker ({attacker_name(att.attacker)}), same outputs",
            f"가장 강한 독립 공격자 ({attacker_name(att.attacker)}), 같은 출력"),
         att.ba, att.ci_low, att.ci_high, C["attacker"]),
    ], 1.0)
    a1.text(0.985, 0.5, tr(f"gap {pp(att.ba - h['ba'])}\n"
                           f"required ≥ {C1_MIN_GAP * 100:.0f} pp",
                           f"간격 {pp(att.ba - h['ba'])}\n"
                           f"요구 ≥ {C1_MIN_GAP * 100:.0f} pp"),
            transform=a1.transAxes, ha="right", va="center", size=NOTE_SIZE + 0.5,
            weight="bold")
    a1.set_title(tr("C1 · Pitfall 1: head vs attacker", "C1 · 함정 1: 식별기 대 공격자"),
                 size=LABEL_SIZE - 0.5)
    decision_badge(a1, dec["C1"])

    # C2: recentred bottleneck vs no defence, with task non-inferiority.
    p0, p2 = d["p0"], d["p2rc"]
    util = dec["C2"]["utility"]
    interval_rows(a2, [
        (tr(f"No defence (P0) · task BA {d['task']['P0']:.3f}",
            f"방어 없음 (P0) · 과제 BA {d['task']['P0']:.3f}"), p0.ba, p0.ci_low, p0.ci_high,
         C["raw"]),
        (tr(f"Recentred bottleneck d = 8 (P2-RC) · task BA {d['task']['P2-RC_d8']:.3f}",
            f"재중심화 병목 d = 8 (P2-RC) · 과제 BA {d['task']['P2-RC_d8']:.3f}"), p2.ba,
         p2.ci_low, p2.ci_high, C["task"]),
    ], 1.0)
    if LANG == "ko":
        a2.set_ylim(-1.85, 2.25)  # room for the two-line note below the bottom row
    a2.text(0.985, 0.04,
            tr(f"identity {pp(p2.ba - p0.ba)} (required ≤ −{C2_MIN_GAP * 100:.0f} pp)\n"
               f"task {pp(util['mean'])}, lower bound {pp(util['lower'])}\n"
               f"(non-inferiority margin {pp(C2_TASK_MARGIN)})",
               f"식별 {pp(p2.ba - p0.ba)} (요구 ≤ −{C2_MIN_GAP * 100:.0f} pp)\n"
               f"과제 {pp(util['mean'])} (하한 {pp(util['lower'])}, "
               f"비열등성 한계 {pp(C2_TASK_MARGIN)})"),
            transform=a2.transAxes, ha="right", va="bottom", size=tr(NOTE_SIZE, 13.5),
            weight="bold", linespacing=tr(1.3, 1.15))
    a2.set_title(tr("C2 · Bottleneck vs no defence", "C2 · 병목 대 방어 없음"),
                 size=LABEL_SIZE - 0.5)
    decision_badge(a2, dec["C2"])

    # C3: commands only, P2-RC d = 8, τ = 0.5, k = 1 vs k = 20.
    k1, k20 = d["c3"]
    interval_rows(a3, [
        (tr("1 command observed", "명령 1개 관찰"), k1.ba, k1.ci_low, k1.ci_high, C["task"]),
        (tr("20 consecutive commands observed", "연속 명령 20개 관찰"), k20.ba, k20.ci_low,
         k20.ci_high, C["task"]),
    ], 0.25)
    threshold_line(a3, C3_FLOOR)
    a3.text(0.985, tr(0.5, 0.56), tr("k = 20 exceeds k = 1, but its\nmargin over 2/54 is not\n"
                           "significant after correction",
                           "k = 20이 k = 1보다 높지만\n2/54 대비 여유는 보정 후\n"
                           "유의하지 않음"),
            transform=a3.transAxes, ha="right", va="center", size=tr(NOTE_SIZE, 14),
            weight="bold", linespacing=1.3)
    a3.set_title(tr("C3 · Commands only (P2-RC, τ = 0.5)", "C3 · 명령만 (P2-RC, τ = 0.5)"),
                 size=LABEL_SIZE - 0.5)
    decision_badge(a3, dec["C3"])

    # C4: published defence (G) vs deployment enrolment, with the reference-session gate.
    ref = d["c4_ref"]
    paper = d["repro"]["paper_accuracy"]
    rows = [(tr(f"Reference-session attacker ({attacker_name(ref.attacker)}): "
                f"{ref.ba:.2%}; paper {paper:.1%}",
                f"기준 세션 공격자 ({attacker_name(ref.attacker)}): "
                f"{ref.ba:.2%}; 논문 {paper:.1%}"), ref.ba, ref.ci_low,
             ref.ci_high, C["head"], paper)]
    for r, n in zip(d["c4"], ("1", "2", "5")):
        label = tr(f"{n} labelled deployment trial{'s' if n != '1' else ''} per user",
                   f"사용자당 배포 기간 라벨 {n}개")
        rows.append((label + (tr(" (tested)", " (검정 대상)") if n == "5" else ""), r.ba,
                     r.ci_low, r.ci_high, C["head"]))
    # Korean: the reference-session gate stands apart from the deployment-enrolment rows.
    ys = interval_rows(a4, rows, 1.0, gap_below=tr(None, 0))
    a4.scatter(paper, ys[0], marker="*", s=230,
               color=C["grl"], edgecolor="black", lw=0.7, zorder=5)
    threshold_line(a4, C4_FLOOR)
    a4.set_title(tr("C4 · Pitfall 2: published defence", "C4 · 공개 방어 (함정 2)"),
                 size=LABEL_SIZE - 0.5)
    decision_badge(a4, dec["C4"])

    if LANG == "ko":
        # One shared label leaves the Korean panels more height for their larger type.
        fig.supxlabel("신원 균형 정확도 (가장 강한 공격자)", y=0.138, size=LABEL_SIZE - 1)
    else:
        for ax in (a3, a4):
            ax.set_xlabel("Identity balanced accuracy (strongest attacker)", size=LABEL_SIZE - 1)
    handles = [
        Line2D([], [], color=C["muted"], lw=3.2, alpha=0.45, marker="o", mfc=C["muted"],
               mec=C["muted"], label=tr("Estimate with 95% participant-bootstrap interval",
                                        "추정값과 95% 참가자 부트스트랩 구간")),
        Line2D([], [], color=C["chance"], ls=":", lw=1.6,
               label=tr("Chance (1/54)", "우연 수준 (1/54)")),
        Line2D([], [], color="black", ls="-.", lw=1.3,
               label=tr("Pre-registered floor (C3: 2/54, C4: 10/54)",
                        "사전등록 기준선 (C3: 2/54, C4: 10/54)")),
        Line2D([], [], marker="*", ls="", ms=14, mfc=C["grl"], mec="black",
               label=tr("Paper's reported result", "논문 보고 결과")),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=tr(4, 2),
               bbox_to_anchor=(0.5, tr(0.045, 0.03)), columnspacing=1.6,
               fontsize=tr(NOTE_SIZE + 0.5, 14.5))
    save(fig, "fig6_lee2019_confirmatory")


# ----------------------------------------------------------------------------
# Figure 7: identity vs number of enrolled users (L, descriptive) and 2a reference
# ----------------------------------------------------------------------------
L_ENROLLED = [9, 18, 27, 54]
N_SERIES = [("P0", "P0", ("No defence (P0)", "방어 없음 (P0)"), C["raw"]),
            ("P2-RC_d8", "P2-RC", ("Recentred bottleneck d = 8 (P2-RC)",
                                   "재중심화 병목 d = 8 (P2-RC)"), C["task"])]


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
        tr("More users lower identity accuracy, but leakage relative to chance grows",
           "사용자가 많을수록 정확도는 낮지만, 우연 대비 누출은 커진다"),
        tr(f"Recentred bottleneck: {p2.ba[9]:.2f} with 9 users ({p2.ba[9] * 9:.1f}× chance), "
           f"{p2.ba[54]:.2f} with 54 users ({p2.ba[54] * 54:.1f}× chance); "
           f"BCI IV 2a's 9 users: {two_a.identity_ba['P2-RC']:.2f}.",
           f"재중심화 병목: 9명 {p2.ba[9]:.2f} (우연의 {p2.ba[9] * 9:.1f}배), "
           f"54명 {p2.ba[54]:.2f} (우연의 {p2.ba[54] * 54:.1f}배); "
           f"BCI IV 2a 9명: {two_a.identity_ba['P2-RC']:.2f}"),
        "DESCRIPTIVE",
        tr("results/exp_l/identity_summary.csv (enrolled subsets, seeds 0–2); "
           "results/exp_b/tradeoff.csv (BCI IV 2a)",
           "results/exp_l/identity_summary.csv (등록 부분집합, seed 0–2); "
           "results/exp_b/tradeoff.csv (BCI IV 2a)"),
    )
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=tr(0.075, 0.085), right=0.985, top=tr(0.80, 0.77),
                        bottom=tr(0.25, 0.28), wspace=tr(0.2, 0.24))
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
        ax.set_xlabel(tr("Users the attacker must tell apart (N)",
                         "공격자가 구별해야 할 사용자 수 (N)"))
    a, b = axes
    grid = pd.Series(range(5, 59), dtype=float)
    a.plot(grid, 1 / grid, color=C["chance"], ls=":", lw=1.6, zorder=1)
    a.text(40, 1 / 40 + 0.015, tr("chance (1/N)", "우연 수준 (1/N)"), size=NOTE_SIZE,
           color=C["chance"])
    a.set_ylim(0, 1.05)
    a.set_ylabel(tr("Identity balanced accuracy\n(strongest attacker)",
                    "신원 균형 정확도 (최강 공격자)"))
    a.set_title(tr("Accuracy", "정확도"))
    b.axhline(1, color=C["chance"], ls=":", lw=1.6, zorder=1)
    b.text(56, 1.6, tr("chance", "우연 수준"), size=NOTE_SIZE, color=C["chance"], ha="right")
    b.set_ylim(0, 60)
    b.set_ylabel(tr("Times chance (accuracy × N)", "우연 대비 배수 (정확도 × N)"))
    b.set_title(tr("Relative to chance", "우연 대비"))
    handles = [Line2D([], [], color=c, lw=2.4, marker="o",
                      label=tr(f"{lbl[0]}, Lee2019 subsets", f"{lbl[1]}, Lee2019 부분집합"))
               for _, _, lbl, c in N_SERIES] + [
        Line2D([], [], color=C["muted"], marker="D", ls="", mfc="white", mew=2, ms=8,
               label=tr("Same model on BCI IV 2a (9 users, experiment B)",
                        "같은 모델, BCI IV 2a (9명, 실험 B)")),
        Line2D([], [], color=C["muted"], lw=8, alpha=0.3,
               label=tr("95% participant-bootstrap interval", "95% 참가자 부트스트랩 구간")),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.045),
               columnspacing=tr(3.0, 2.0), fontsize=tr(NOTE_SIZE + 0.5, 14.5))
    save(fig, "fig7_lee2019_users")


# ----------------------------------------------------------------------------
# Figure 8: experiment K (exploratory), identity in decoded command streams
# ----------------------------------------------------------------------------
# K's declared headline setting (configs/exp_k.yaml decision_rule): Lee2019, P2-RC_d8,
# tau = 0.8, k* = 20, g* = 3, EQ variant confusion, cross-fitted enrolment primary.
K_HEAD = dict(dataset="lee2019", condition="P2-RC_d8", tau=0.8, k=20, g=3)
K_DATASETS = [("lee2019", "Lee2019"), ("bnci2014_001", "BCI IV 2a")]
K_CONDITIONS = [("P2-RC_d8", ("Recentred bottleneck d = 8 (P2-RC), headline",
                              "재중심화 병목 d = 8 (P2-RC), 대표 설정"), C["task"], 2.8),
                ("P0", ("No defence (P0)", "방어 없음 (P0)"), C["raw"], 1.6)]
K_SCENARIOS = [
    ("within_day", "-", "o",
     ("Within-day: enrol and attack within the deployment day (A → B and B → A averaged)",
      "같은 날 (within-day): 배포일 안에서 등록 → 공격 (A ↔ B 양방향 평균)")),
    ("cross_day", "--", "s",
     ("Cross-day: enrol on the calibration day (cross-fitted), attack the deployment day",
      "다른 날 (cross-day): 보정일 등록 (교차 적합) → 배포일 공격")),
]


def load_k_curves():
    """Undefended N-way rows of identity_curves.csv with family_set = bayes, i.e. the strongest
    of the four Bayes attackers per point (beat/exp_k.py ``strongest``). frontier_table builds
    frontier.csv's nway/excess columns from exactly these rows, so the values coincide."""
    c = pd.read_csv(RES / "exp_k" / "identity_curves.csv",
                    usecols=["dataset", "condition", "tau", "defence", "scenario", "family_set",
                             "metric", "k", "ba", "chance", "users", "seeds", "candidates"])
    return c[(c.family_set == "bayes") & (c.defence == "none") & (c.metric == "nway")
             & (c.tau == K_HEAD["tau"]) & c.scenario.isin([s for s, *_ in K_SCENARIOS])
             & c.condition.isin([k for k, *_ in K_CONDITIONS])]


def fig_k_commands():
    c = load_k_curves()
    k = K_HEAD["k"]
    head = c[(c.dataset == K_HEAD["dataset"]) & (c.condition == K_HEAD["condition"])]
    seeds, attackers = int(head.seeds.iloc[0]), int(head.candidates.iloc[0])
    ba = head.set_index(["scenario", "k"]).ba
    kmax = head.groupby("scenario").k.max()
    wd = ba[("within_day", k)], ba[("within_day", kmax["within_day"])]
    cd = ba[("cross_day", k)], ba[("cross_day", kmax["cross_day"])]
    tau = K_HEAD["tau"]
    fig = new_figure(
        tr("Exploratory K: decoded command streams reveal the user as commands accumulate",
           "탐색적 K: 해독된 명령이 쌓일수록 사용자가 드러난다"),
        tr(f"Lee2019, P2-RC, τ = {tau:g}: within-day {wd[0]:.3f} (k = {k}) → {wd[1]:.3f} "
           f"(k = {kmax['within_day']}); cross-day {cd[0]:.3f} (k = {k}) → {cd[1]:.3f} "
           f"(k = {kmax['cross_day']}).\nPer point, the strongest of {attackers} Bayes attackers "
           "(categorical or Markov, target-blind or target-aware: cat_*, markov_*), as in "
           f"frontier.csv; mean of {seeds} seeds.",
           f"Lee2019, P2-RC, τ = {tau:g}: 같은 날 {wd[0]:.3f} (k = {k}) → {wd[1]:.3f} "
           f"(k = {kmax['within_day']}), 다른 날 {cd[0]:.3f} (k = {k}) → {cd[1]:.3f} "
           f"(k = {kmax['cross_day']})\nBayes 공격자 {attackers}종 (범주형·Markov, 목표 "
           f"미인지·인지: cat_*, markov_*) 중 최강 (frontier.csv 정의), seed {seeds}개 평균"),
        "EXPLORATORY",
        f"results/exp_k/identity_curves.csv (family_set = bayes, defence = none, metric = nway, "
        f"τ = {tau:g})",
    )
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=tr(0.075, 0.105), right=0.985, top=tr(0.78, 0.735),
                        bottom=tr(0.27, 0.345), wspace=tr(0.17, 0.2))
    for ax, (dataset, name) in zip(axes, K_DATASETS):
        part = c[c.dataset == dataset]
        users = int(part.users.iloc[0])
        ax.axvline(k, color="#BBBBBB", lw=1.2, zorder=0)
        for cond, _, colour, lw in K_CONDITIONS:
            for scen, ls, marker, _ in K_SCENARIOS:
                r = part[(part.condition == cond) & (part.scenario == scen)].sort_values("k")
                ax.plot(r.k, r.ba, ls=ls, marker=marker, color=colour, lw=lw, ms=7,
                        zorder=4 if cond == K_HEAD["condition"] else 3)
                if dataset != K_HEAD["dataset"] or cond != K_HEAD["condition"]:
                    continue
                # The values the talk quotes: k* and kmax of the headline cell.
                within = scen == "within_day"
                for kk in (k, int(r.k.max())):
                    y = r.set_index("k").ba[kk]
                    ax.annotate(f"{y:.3f}", (kk, y), xytext=(0, 9 if within else -10),
                                textcoords="offset points", ha="center",
                                va="bottom" if within else "top", size=tr(NOTE_SIZE, 13.5),
                                weight="bold", color=colour, zorder=6,
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                                          alpha=0.85))
        ax.axhline(1 / users, color=C["chance"], ls=":", lw=1.6, zorder=1)
        ax.axhline(2 / users, color=C["chance"], ls="--", lw=1.2, zorder=1)
        ks = sorted(part.k.unique())
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_ylim(0, 1.15 * part.ba.max())
        ax.set_title(tr(f"{name} ({users} users)", f"{name} ({users}명)"), pad=6)
        ax.set_xlabel(tr("Commands observed from one user (k, log scale)",
                         "한 사용자에게서 관찰한 명령 수 (k, 로그 축)"))
    axes[0].set_ylabel(tr("Identification among all N users\n(N-way balanced accuracy)",
                          "신원 식별 균형 정확도 (전체 N명 중)"), **tr({}, {"size": 15}))
    handles = [Line2D([], [], color=colour, lw=lw, label=tr(*lbl))
               for _, lbl, colour, lw in K_CONDITIONS]
    handles += [Line2D([], [], color=C["chance"], ls=":", lw=1.6,
                       label=tr("Chance (1/N)", "우연 수준 (1/N)")),
                Line2D([], [], color=C["chance"], ls="--", lw=1.4,
                       label=tr("Twice chance (2/N)", "우연의 2배 (2/N)"))]
    handles += [Line2D([], [], color="black", ls=ls, marker=marker, lw=2, ms=7, label=tr(*lbl))
                for _, ls, marker, lbl in K_SCENARIOS]
    handles.append(Line2D([], [], color="#BBBBBB", lw=1.2,
                          label=tr(f"Declared k* = {k} (configs/exp_k.yaml); labels: headline "
                                   "values at k* and kmax",
                                   f"선언된 k* = {k} (configs/exp_k.yaml); 숫자: k*와 kmax의 "
                                   "대표 설정 값")))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.05),
               columnspacing=2.0, fontsize=tr(NOTE_SIZE + 0.5, 14.5))
    save(fig, "fig8_k_command_streams")


# ----------------------------------------------------------------------------
# Figure 9: experiment K (exploratory), randomized response vs group-equalized decoding
# ----------------------------------------------------------------------------
K_GROUPS = [(2, "#E69F00"), (3, "#0072B2"), (5, "#009E73")]
K_RR = ["rr_q0.1", "rr_q0.25", "rr_q0.5", "rr_q0.75"]
K_EQ = [("eqm", "^"), ("eqc", "v")]
K_FRONTIER_SCENARIOS = [
    ("within_day", ("Within-day: deployment day, A → B and B → A averaged",
                    "같은 날 (within-day): 배포일 A ↔ B 양방향 평균")),
    ("cross_day", ("Cross-day: calibration → deployment day (filled: cross-fitted)",
                   "다른 날 (cross-day): 보정일 → 배포일 (채움: 교차 적합)")),
]


def load_k_frontier():
    """frontier.csv rows of the headline cell, plus its N-way user count and chance from
    identity_curves.csv (frontier.csv has neither column)."""
    f = pd.read_csv(RES / "exp_k" / "frontier.csv")
    sel = f[(f.dataset == K_HEAD["dataset"]) & (f.condition == K_HEAD["condition"])
            & (f.tau == K_HEAD["tau"])]
    c = load_k_curves()
    cell = c[(c.dataset == K_HEAD["dataset"]) & (c.condition == K_HEAD["condition"])]
    return sel.set_index("defence"), int(cell.users.iloc[0]), float(cell.chance.iloc[0])


def eq_point(ax, x, y, y_insample, marker, colour, size=120, alpha=1.0):
    """EQ operating point; for cross-day also the in-sample enrolment value (hollow)."""
    if y_insample is not None:
        ax.plot([x, x], [y, y_insample], color=colour, lw=1, alpha=alpha, zorder=4)
        ax.scatter(x, y_insample, marker=marker, s=size, facecolor="white", edgecolor=colour,
                   lw=1.6, alpha=alpha, zorder=5)
    ax.scatter(x, y, marker=marker, s=size, color=colour, edgecolor="black", lw=0.6,
               alpha=alpha, zorder=6)


def value_label(ax, x, y, offset, ha, va, colour):
    ax.annotate(f"{y:.3f}", (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va,
                size=tr(NOTE_SIZE, 13.5), weight="bold", color=colour, zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


def fig_k_frontier():
    f, n_users, chance = load_k_frontier()
    k, g_star, tau = K_HEAD["k"], K_HEAD["g"], K_HEAD["tau"]
    u = f.weighted_utility
    w = f.wrong_weight.iloc[0]
    path = ["none"] + K_RR
    eq_star = f"eqc_g{g_star}"
    ex = {s: f[f"g{g_star}_excess_{s}_k{k}"] for s, _ in K_FRONTIER_SCENARIOS}
    gall = f.loc["eqc_gall"]
    fig = new_figure(
        tr("Exploratory K: randomized response vs group-equalized decoding (headline setting)",
           "탐색적 K: 무작위 응답(RR) 대 그룹 균등화 해독(EQ), 대표 설정"),
        tr(f"g = {g_star}, EQ-confusion: within-group excess within-day "
           f"{ex['within_day']['none']:.3f} → {ex['within_day'][eq_star]:.3f}, cross-day "
           f"{ex['cross_day']['none']:.3f} → {ex['cross_day'][eq_star]:.3f}; weighted utility "
           f"{u['none']:.3f} → {u[eq_star]:.3f}.\nLee2019 ({n_users} users), recentred bottleneck "
           f"P2-RC, τ = {tau:g} (w = {w:g}), k = {k}; strongest of 4 Bayes attackers.",
           f"g = {g_star}, EQ-confusion: 그룹 내 초과 같은 날 {ex['within_day']['none']:.3f} → "
           f"{ex['within_day'][eq_star]:.3f}, 다른 날 {ex['cross_day']['none']:.3f} → "
           f"{ex['cross_day'][eq_star]:.3f}; 가중 유용성 {u['none']:.3f} → {u[eq_star]:.3f}\n"
           f"Lee2019 ({n_users}명), 재중심화 병목 P2-RC, τ = {tau:g} (w = {w:g}), k = {k}, "
           "Bayes 공격자 4종 중 최강"),
        "EXPLORATORY",
        f"results/exp_k/frontier.csv (lee2019, {K_HEAD['condition']}, τ = {tau:g}; k = {k} "
        "columns); users and chance: results/exp_k/identity_curves.csv",
    )
    axes = fig.subplots(2, 2, sharex=True, sharey="row")
    fig.subplots_adjust(left=tr(0.09, 0.1), right=0.985, top=tr(0.78, 0.76),
                        bottom=tr(0.24, 0.28), wspace=0.06, hspace=0.12)
    for col, (scen, title) in enumerate(K_FRONTIER_SCENARIOS):
        top, bottom = axes[:, col]
        cross = scen == "cross_day"
        for g, colour in K_GROUPS:
            head = g == g_star  # the declared g* is drawn bold, the others muted
            lw, ms, size, alpha = (3.0, 7, 170, 1.0) if head else (1.4, 5, 100, 0.55)
            y = f[f"g{g}_excess_{scen}_k{k}"]
            top.plot(u[path], y[path], "-o", color=colour, lw=lw, ms=ms, alpha=alpha,
                     zorder=4 if head else 3)
            top.scatter(u["none"], y["none"], marker="*", s=380 if head else 240, color=colour,
                        edgecolor="black", lw=0.6, alpha=alpha, zorder=5)
            for kind, marker in K_EQ:
                d = f"{kind}_g{g}"
                eq_point(top, u[d], y[d],
                         f.loc[d, f"g{g}_excess_cross_day_eq_insample_k{k}"] if cross else None,
                         marker, colour, size, alpha)
                eq_point(bottom, u[d], f.loc[d, f"nway_ba_{scen}_k{k}"],
                         f.loc[d, f"nway_ba_cross_day_eq_insample_k{k}"] if cross else None,
                         marker, colour, size, alpha)
            if head:  # undefended value to the right, clear of the muted RR points on the left
                value_label(top, u["none"], y["none"], (14, 6), "left", "center", colour)
                # Clear of the EQ-marginal triangle just above and to the right.
                value_label(top, u[eq_star], y[eq_star], (17, -3), "left", "center", colour)
        nway = f[f"nway_ba_{scen}_k{k}"]
        bottom.plot(u[path], nway[path], "-o", color="black", lw=1.8, ms=6, zorder=3)
        bottom.scatter(u["none"], nway["none"], marker="*", s=300, color="black", zorder=5)
        # Within-day the undefended value goes right: on its left sits the q = 0.1 label.
        value_label(bottom, u["none"], nway["none"], (-12, 6) if cross else (14, 6),
                    "right" if cross else "left", "bottom" if cross else "center", "black")
        eq_point(bottom, u["eqc_gall"], gall[f"nway_ba_{scen}_k{k}"],
                 gall[f"nway_ba_cross_day_eq_insample_k{k}"] if cross else None, "v", C["muted"])
        if not cross:  # q is the same in both columns; labelled once, above the points
            bottom.annotate(tr("g = all", "g = 전체"),
                            (u["eqc_gall"], gall[f"nway_ba_{scen}_k{k}"]), xytext=(9, 5),
                            textcoords="offset points", ha="left", va="bottom",
                            size=tr(NOTE_SIZE, 13.5), color=C["muted"])
            for d in K_RR:
                bottom.annotate(f"q = {d[4:]}", (u[d], nway[d]), xytext=(0, 9),
                                textcoords="offset points", ha="center", va="bottom",
                                size=tr(NOTE_SIZE, 13.5), color=C["muted"])
        else:  # the g = all note once, in the column whose upper left is empty
            bottom.text(0.3, 0.6,
                        tr(f"EQ-confusion, g = all (grey ▼):\nutility {gall.weighted_utility:.3f}, "
                           f"hold {gall.hold_rate:.3f}",
                           f"EQ-confusion, g = 전체 (회색 ▼):\n유용성 "
                           f"{gall.weighted_utility:.3f}, 정지 비율 {gall.hold_rate:.3f}"),
                        transform=bottom.transAxes, ha="center", va="bottom",
                        size=tr(NOTE_SIZE, 13.5), color=C["muted"])
        top.axhline(0, color=C["chance"], ls=":", lw=1.6, zorder=1)
        bottom.axhline(chance, color=C["chance"], ls=":", lw=1.6, zorder=1)
        top.set_title(tr(*title), pad=6, **tr({}, {"size": 15}))
        bottom.set_xlabel(tr(f"Weighted utility P(correct) − w·P(wrong), w = τ/(1 − τ) = {w:g}",
                             f"가중 유용성 P(정답) − w·P(오답), w = τ/(1 − τ) = {w:g}"))
    a_top, a_bottom = axes[:, 0]
    if LANG == "ko":  # keeps the first x tick label off the "0.0" y tick label
        a_bottom.tick_params(axis="x", pad=7)
    excess = pd.concat([f[f"g{g}_excess_{s}_k{k}"] for g, _ in K_GROUPS for s, _ in
                        K_FRONTIER_SCENARIOS])
    shown = u[path + [f"{kind}_g{g}" for kind, _ in K_EQ for g, _ in K_GROUPS] + ["eqc_gall"]]
    a_top.set_ylim(min(0, excess.min()) - 0.02, 1.15 * excess.max())
    a_top.set_xlim(shown.min() - 0.08, shown.max() + 0.12)
    a_top.text(0.985, 0, tr("chance within the group (excess = accuracy − 1/g = 0)",
                            "그룹 내 우연 수준 (초과 = 정확도 − 1/g = 0)"),
               transform=a_top.get_yaxis_transform(), ha="right", va="bottom",
               size=tr(NOTE_SIZE, 13.5), color=C["chance"])
    a_bottom.text(0.2, chance, tr(f"chance (1/{n_users})", f"우연 수준 (1/{n_users})"),
                  transform=a_bottom.get_yaxis_transform(), ha="left", va="bottom",
                  size=tr(NOTE_SIZE, 13.5), color=C["chance"])
    nway_all = pd.concat([f[f"nway_ba_{s}_k{k}"] for s, _ in K_FRONTIER_SCENARIOS])
    a_bottom.set_ylim(0, 1.2 * nway_all.max())  # headroom for the undefended value label
    a_top.set_ylabel(tr("Within-group excess\n(accuracy − 1/g)", "그룹 내 초과 식별"),
                     size=tr(LABEL_SIZE, 15))
    a_bottom.set_ylabel(tr(f"Identification among\nall {n_users} users (N-way)",
                           f"전체 {n_users}명 중 식별"), size=tr(LABEL_SIZE, 15))
    handles = [
        Line2D([], [], marker="*", ls="", color=C["muted"], ms=15,
               label=tr("No defence", "방어 없음")),
        Line2D([], [], marker="o", color=C["muted"], lw=1.8, ms=6,
               label=tr("Randomized response RR (q = 0.1–0.75)", "무작위 응답 RR (q = 0.1–0.75)")),
        Line2D([], [], marker="s", ls="", color="black", ms=10,
               label=tr("Black: no group (N-way, lower row)", "검정: 그룹 없음 (아래 줄, 전체 식별)")),
        Line2D([], [], marker="^", ls="", color=C["muted"], ms=10,
               label=tr("EQ-marginal: equal P(output) in a group",
                        "EQ-marginal: 그룹 내 P(출력) 균등화")),
        Line2D([], [], marker="v", ls="", color=C["muted"], ms=10,
               label=tr("EQ-confusion: equal P(output | target) in a group",
                        "EQ-confusion: 그룹 내 P(출력 | 목표) 균등화")),
        Line2D([], [], marker="v", ls="", mfc="white", mec=C["muted"], mew=1.6, ms=10,
               label=tr("Hollow: cross-day, in-sample enrolment (lower bound)",
                        "속 빈 표시: 다른 날, 표본 내 등록 (하한)")),
    ] + [Line2D([], [], marker="s", ls="", color=colour, ms=10,
                label=tr(f"Group size g = {g}" + (" (declared g*, bold)" if g == g_star else ""),
                         f"그룹 크기 g = {g}" + (" (선언된 g*, 굵게)" if g == g_star else "")))
         for g, colour in K_GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, tr(0.05, 0.042)),
               columnspacing=tr(2.0, 1.0), handletextpad=0.4, fontsize=tr(NOTE_SIZE + 0.5, 14))
    save(fig, "fig9_k_rr_eq_frontier")


FIGURES = [fig_pitfall1, fig_frontier, fig_commands, fig_enrolment, fig_timeline, fig_lee2019,
           fig_crowd, fig_k_commands, fig_k_frontier]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the summary figures.")
    parser.add_argument("--lang", choices=("en", "ko", "all"), default="all",
                        help="label language: English, Korean or both (default)")
    args = parser.parse_args(argv)
    for lang in ("en", "ko") if args.lang == "all" else (args.lang,):
        set_language(lang)
        for make in FIGURES:
            make()
        for p in sorted(OUT.glob("*")):
            print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
