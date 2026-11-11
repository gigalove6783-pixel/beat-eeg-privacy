"""Draw the figures of the v3 talk deck from the audit result files.

Every plotted value is read from ``results/audit/``; no number is typed in. Each figure is drawn
at the size it occupies on its slide, so one figure point is one slide point and the label sizes
set here are the sizes the audience sees. Sources and tiers are written on the slides.

Output: ``presentation/figures/v3/*.png`` (Korean labels).
Run from the project root: ``python scripts/make_v3_figures.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "audit"
OUT = ROOT / "presentation" / "figures" / "v3"

# The deck is 1920 px wide on a 13.333 in slide: 144 slide pixels per inch.
SLIDE_PX_PER_INCH = 144
DPI = 2 * SLIDE_PX_PER_INCH

BLUE, VERMILION, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GRAY, NAVY, INK, MUTED, LIGHT = "#8A939E", "#14202E", "#1B1F24", "#5B6470", "#E3E8EE"
KS = [1, 5, 10, 20, 50, 100]
DETECTED = "linkage detected"

RUNG_LABELS = {
    "features_P0": "전체 특징",
    "features_P2-RC_d8": "8차원 병목 특징",
    "commands_shared_tau0.50": "공용 해독기 명령 τ 0.50",
    "commands_shared_tau0.60": "공용 해독기 명령 τ 0.60",
    "commands_shared_tau0.80": "공용 해독기 명령 τ 0.80",
    "commands_per_user_tau0.50": "사용자별 해독기 명령 τ 0.50",
    "commands_per_user_tau0.60": "사용자별 해독기 명령 τ 0.60",
    "commands_per_user_tau0.80": "사용자별 해독기 명령 τ 0.80",
    "commands_P2-RC_d8_tau0.80": "명령 τ 0.80",
}
COMPARATOR_NAMES = {
    "cosine": "코사인",
    "lda_cosine": "LDA + 코사인",
    "pair_logistic": "쌍 로지스틱",
    "cosine_rich": "코사인",
    "lda_cosine_rich": "LDA + 코사인",
}
COMPARATOR_LABELS = {
    "cosine": "코사인 · 평균",
    "lda_cosine": "LDA + 코사인 · 평균",
    "cosine_rich": "코사인 · 풍부",
    "lda_cosine_rich": "LDA + 코사인 · 풍부",
}


def style() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "Noto Sans KR", "sans-serif"],
        "axes.unicode_minus": False,
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": LIGHT,
        "grid.linewidth": 1.0,
        "savefig.facecolor": "white",
    })


def figure(width_px: int, height_px: int, **kwargs):
    """A figure the size it is placed on the slide, given in slide pixels."""
    return plt.subplots(figsize=(width_px / SLIDE_PX_PER_INCH, height_px / SLIDE_PX_PER_INCH),
                        **kwargs)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {(OUT / name).relative_to(ROOT).as_posix()}")


def lower_limit(values: pd.Series, floor: float = 0.45) -> float:
    """Axis start: ``floor``, or the 0.05 step below the lowest plotted value, so that no
    interval is cut off."""
    low = float(np.nanmin(values.to_numpy(dtype=float)))
    return min(floor, float(np.floor((low - 0.01) / 0.05)) * 0.05)


def chance_line(ax: plt.Axes, vertical: bool = False, label_at: float | None = None) -> None:
    if vertical:
        ax.axvline(0.5, color=GRAY, ls=":", lw=2, zorder=1)
        if label_at is not None:
            ax.text(0.506, label_at, "우연 0.5", color=MUTED, ha="left", va="center")
    else:
        ax.axhline(0.5, color=GRAY, ls=":", lw=2, zorder=1)


def log_k_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(KS))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(KS[0] / 1.25, KS[-1] * 1.25)


def coloured_ticks(ax: plt.Axes, ticks: list[float], labels: list[str], colours: list[str]) -> None:
    ax.set_yticks(ticks, labels)
    for tick, colour in zip(ax.get_yticklabels(), colours):
        tick.set_color(colour)


# ------------------------------------------------------------------ curves over k

def linkage_over_k(cohort: str, rungs: list[tuple[str, str]], name: str,
                   size: tuple[int, int]) -> None:
    """Strongest comparator per cell, same-day (solid) beside cross-day (dashed)."""
    summary = pd.read_csv(AUDIT / "summary.csv")
    cells = summary[(summary.cohort == cohort) & summary.strongest & (summary.status == "ok")
                    & summary.rung.isin([rung for rung, _ in rungs])]
    hollow = bool((cells.wording != DETECTED).any())
    fig, ax = figure(*size)
    fig.subplots_adjust(left=0.09, right=0.67, top=0.97, bottom=0.16)
    for rung, colour in rungs:
        for condition, ls, marker in (("same-day", "-", "o"), ("cross-day", "--", "s")):
            d = cells[(cells.rung == rung) & (cells.condition == condition)].sort_values("k")
            ax.fill_between(d.k, d.ci_low, d.ci_high, color=colour, alpha=0.10, lw=0)
            ax.plot(d.k, d.auc, ls=ls, color=colour, lw=2.2, zorder=3)
            for _, row in d.iterrows():
                filled = row.wording == DETECTED
                ax.plot(row.k, row.auc, marker=marker, ms=7, mew=1.8, color=colour,
                        mfc=colour if filled else "white", zorder=4)
    chance_line(ax)
    log_k_axis(ax)
    ax.set_ylim(lower_limit(pd.concat([cells.ci_low, cells.auc])), 1.02)
    ax.set_xlabel("한 사람에게서 모은 출력 수 k (로그 축)")
    ax.set_ylabel("연결 AUC")
    handles = [Line2D([], [], color=c, lw=3, label=RUNG_LABELS[r]) for r, c in rungs]
    handles += [
        Line2D([], [], color=INK, lw=2, ls="-", marker="o", ms=7, label="같은 날"),
        Line2D([], [], color=INK, lw=2, ls="--", marker="s", ms=7, label="다른 날"),
        Line2D([], [], color=GRAY, lw=2, ls=":", label="우연 0.5"),
        Patch(color=GRAY, alpha=0.25, label="띠: 95% 구간"),
    ]
    if hollow:
        handles.append(Line2D([], [], color=INK, lw=0, marker="o", ms=7, mew=1.8, mfc="white",
                              label="속 빈 점:\n검출 안 됨"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              handlelength=2.0, labelspacing=0.6)
    save(fig, name)


# ------------------------------------------------------------------ weak attacker

def weak_attacker(size: tuple[int, int]) -> None:
    """Lee2019, 8-d bottleneck, cross-day, k = 20: mean-summary and rich-summary comparators
    beside the commands decoded from the same features."""
    summary = pd.read_csv(AUDIT / "summary.csv")
    cell = summary[(summary.cohort == "lee2019") & (summary.condition == "cross-day")
                   & (summary.k == 20) & (summary.status == "ok")]
    feat = cell[cell.rung == "features_P2-RC_d8"].set_index("comparator")
    commands = cell[cell.rung.str.startswith("commands")].sort_values("tau")
    groups = [
        (GRAY, [(f"평균 요약 · {COMPARATOR_NAMES[c]}", feat.loc[c])
                for c in ("cosine", "lda_cosine", "pair_logistic")]),
        (VERMILION, [(f"명령 τ {r.tau:.2f} · 베이즈 인자", r) for _, r in commands.iterrows()]),
        (GREEN, [(f"풍부한 요약 · {COMPARATOR_NAMES[c]}", feat.loc[c])
                 for c in ("cosine_rich", "lda_cosine_rich")]),
    ]
    fig, ax = figure(*size)
    fig.subplots_adjust(left=0.37, right=0.98, top=0.97, bottom=0.17)
    y, ticks, labels, colours, lows, gaps = 0.0, [], [], [], [], []
    for colour, items in groups:
        for label, r in items:
            ax.plot([r.ci_low, r.ci_high], [y, y], color=colour, lw=3, solid_capstyle="butt")
            ax.plot(r.auc, y, "o", ms=10, color=colour, zorder=4)
            ticks.append(y)
            labels.append(label)
            colours.append(colour)
            lows.append(r.ci_low)
            y += 1.0
        gaps.append(y - 0.25)
        y += 0.5
    command_best = commands.auc.max()
    ax.axvline(command_best, color=VERMILION, ls="--", lw=1.6, zorder=1)
    ax.text(command_best + 0.006, -0.45, "명령 최고값 =\n특징 최선 공격의\n하한",
            color=VERMILION, fontsize=16, ha="left", va="top", linespacing=1.15)
    chance_line(ax, vertical=True, label_at=gaps[0])
    coloured_ticks(ax, ticks, labels, colours)
    ax.set_ylim(ticks[-1] + 0.6, -0.6)
    ax.set_xlim(lower_limit(pd.Series(lows)), 0.84)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("다른 날 연결 AUC, 출력 20개 (선: 95% 구간)")
    save(fig, "v3_fig2_lee2019_weak_attacker.png")


# ------------------------------------------------------------------ ladders

def ladder_rows(cohort: str, rungs: list[str]) -> pd.DataFrame:
    ladder = pd.read_csv(AUDIT / "ladder.csv")
    rows = ladder[ladder.cohort == cohort].set_index("rung")
    return rows.loc[rungs]


def yang2025_ladder(size: tuple[int, int]) -> None:
    """Cross-day linkage on the days no model was fitted on, grouped by whether the task BA
    matches: the shared-decoder commands beside the 8-d features they are decoded from."""
    groups = [("같은 BA · 맞춘 비교", NAVY,
               ["features_P2-RC_d8", "commands_shared_tau0.50", "commands_shared_tau0.60"]),
              ("BA 다름 · 맞춘 비교 아님", GRAY, ["features_P0", "commands_per_user_tau0.50"])]
    rows = ladder_rows("yang2025", [rung for _, _, rungs in groups for rung in rungs])
    if rows.loc[groups[0][2], "task_ba"].round(3).nunique() != 1:
        raise ValueError("the matched rungs differ in task BA")
    fig, ax = figure(*size)
    fig.subplots_adjust(left=0.40, right=0.86, top=0.97, bottom=0.17)
    side = ax.get_yaxis_transform()
    ax.text(1.03, 0.0, "과제 BA", transform=side, ha="left", va="center", fontweight="bold")
    y, ticks, labels, colours = 0.0, [], [], []
    for header, colour, rungs in groups:
        ax.text(-0.02, y, header, transform=side, ha="right", va="center", fontweight="bold",
                color=colour)
        y += 1.0
        for rung in rungs:
            r = rows.loc[rung]
            ax.plot([r.cross_day_auc_low, r.cross_day_auc_high], [y, y], color=colour, lw=3,
                    solid_capstyle="butt")
            ax.plot(r.cross_day_auc, y, "s", ms=10, color=colour, zorder=4)
            ax.text(1.03, y, f"{r.task_ba:.3f}", transform=side, ha="left", va="center",
                    color=colour)
            ticks.append(y)
            labels.append(RUNG_LABELS[rung])
            colours.append(colour)
            y += 1.0
        y += 0.4
    chance_line(ax, vertical=True, label_at=0.0)
    coloured_ticks(ax, ticks, labels, colours)
    ax.set_ylim(ticks[-1] + 0.5, -0.5)
    ax.set_xlim(lower_limit(rows.cross_day_auc_low), 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("다른 날 연결 AUC, 출력 20개 (선: 95% 구간)")
    save(fig, "v3_fig3_yang2025_ladder.png")


def ladders_lee2019_2a(size: tuple[int, int]) -> None:
    rungs = ["features_P0", "features_P2-RC_d8", "commands_shared_tau0.50",
             "commands_shared_tau0.80", "commands_per_user_tau0.50",
             "commands_per_user_tau0.60", "commands_per_user_tau0.80"]
    panels = (("lee2019", "Lee2019 54명"), ("bnci2014_001", "BCI IV 2a 9명"))
    data = {cohort: ladder_rows(cohort, rungs) for cohort, _ in panels}
    low = lower_limit(pd.concat([d.same_day_auc_low for d in data.values()]))
    fig, axes = figure(*size, ncols=2, sharey=True)
    fig.subplots_adjust(left=0.26, right=0.985, top=0.91, bottom=0.30, wspace=0.08)
    for ax, (cohort, title) in zip(axes, panels):
        for i, (_, r) in enumerate(data[cohort].iterrows()):
            ax.plot([r.same_day_auc_low, r.same_day_auc_high], [i, i], color=NAVY, lw=3,
                    solid_capstyle="butt")
            filled = r.same_day_wording == DETECTED
            ax.plot(r.same_day_auc, i, "o", ms=10, mew=2, color=NAVY,
                    mfc=NAVY if filled else "white", zorder=4)
        ax.axhline(1.5, color=MUTED, lw=1.2)
        chance_line(ax, vertical=True)
        ax.set_yticks(range(len(rungs)), [RUNG_LABELS[r] for r in rungs])
        ax.set_ylim(len(rungs) - 0.45, -0.55)
        ax.set_xlim(low, 1.02)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("같은 날 연결 AUC, 출력 20개")
    handles = [Line2D([], [], color=NAVY, lw=3, marker="o", ms=10, label="검출 (선: 95% 구간)"),
               Line2D([], [], color=NAVY, lw=3, marker="o", ms=10, mfc="white", mew=2,
                      label="검출 안 됨 (증거 부족, 없음 아님)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=2,
               frameon=False)
    save(fig, "v3_fig7_ladders_lee2019_2a.png")


# ------------------------------------------------------------------ validation

def validation(size: tuple[int, int]) -> None:
    v = json.loads((AUDIT / "validation.json").read_text(encoding="utf-8"))
    shapes = {(c["users"], c["days"], c["effect"]): c for c in v["whole_procedure"]}
    fig, axes = figure(*size, ncols=3, gridspec_kw={"width_ratios": [1, 1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.2, wspace=0.38)

    ax = axes[0]
    names = [(9, 2), (54, 2), (51, 3)]
    rates = [shapes[(u, d, 0.0)]["rejection_rate"] for u, d in names]
    ax.bar(range(3), rates, color=BLUE, width=0.6)
    ax.axhline(0.05, color=VERMILION, ls="--", lw=2)
    ax.text(2.45, 0.051, "명목 수준", color=VERMILION, ha="right", va="bottom")
    ax.set_xticks(range(3), [f"{u}명\n{d}일" for u, d in names])
    ax.set_ylim(0, 0.07)
    ax.set_title("거짓 검출률\n(신원 정보 없음)")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    gated = [(54, 2), (51, 3)]
    for j, (key, label, colour, marker) in enumerate((
            ("auc_coverage", "평균 AUC", BLUE, "o"),
            ("contrast_coverage", "같은 날 − 다른 날", NAVY, "D"))):
        vals = [shapes[(u, d, 2.0)][key] for u, d in gated]
        ax.plot([i + (j - 0.5) * 0.3 for i in range(2)], vals, marker, ms=11, color=colour,
                label=label, zorder=4)
    low, high = shapes[(54, 2, 2.0)]["coverage_requirement"]
    ax.axhspan(low, high, color=GREEN, alpha=0.12, lw=0)
    ax.text(1.55, low + 0.003, "요구 범위", color=GREEN, ha="right", va="bottom")
    ax.set_xticks(range(2), [f"{u}명\n{d}일" for u, d in gated])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.85, 1.0)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00], ["0.85", "0.90", "0.95", "1.00"])
    ax.set_title("95% 구간의 참값 포함률\n(심은 효과)")
    ax.legend(loc="lower center", ncol=2, frameon=False, handlelength=1.0, columnspacing=1.0)
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    extra = [a for a in v["additional_null_validation"]
             if a["configuration"]["simulations_per_case"] == 200][0]
    cases = extra["whole_procedure"]
    counts = [c["rejections"] for c in cases]
    lo, hi = cases[0]["binomial_99_count_bounds"]
    ax.axhspan(lo, hi, color=GREEN, alpha=0.12, lw=0)
    ax.bar(range(len(cases)), counts, color=BLUE, width=0.6)
    nominal = 0.05 * cases[0]["simulations"]
    ax.axhline(nominal, color=VERMILION, ls="--", lw=2)
    ax.text(2.45, hi - 0.4, "99% 허용", color=GREEN, ha="right", va="top")
    ax.text(-0.45, nominal + 0.3, "명목 수준", color=VERMILION, ha="left", va="bottom")
    ax.set_xticks(range(len(cases)), ["코사인\n풍부", "LDA\n풍부", "베이즈\n인자"])
    ax.set_ylim(0, hi + 3)
    ax.set_title("확장 영가설 실행\n(51명×3일, 거짓 검출 횟수)")
    ax.grid(axis="x", visible=False)
    save(fig, "v3_fig4_validation.png")


# ------------------------------------------------------------------ comparators (backup)

def yang2025_comparators(size: tuple[int, int]) -> None:
    y = pd.read_csv(AUDIT / "yang2025" / "summary.csv")
    colours = {"cosine": GRAY, "lda_cosine": ORANGE, "cosine_rich": BLUE, "lda_cosine_rich": GREEN}
    d = y[(y.rung == "features_P0") & (y.condition == "cross-day") & (y.status == "ok")
          & y.comparator.isin(colours)]
    fig, ax = figure(*size)
    fig.subplots_adjust(left=0.09, right=0.66, top=0.97, bottom=0.16)
    for comp, colour in colours.items():
        c = d[d.comparator == comp].sort_values("k")
        ax.fill_between(c.k, c.ci_low, c.ci_high, color=colour, alpha=0.08, lw=0)
        ax.plot(c.k, c.auc, "-o", color=colour, lw=2.2, ms=7, label=COMPARATOR_LABELS[comp])
    v1 = d[d.role == "V1"].iloc[0]
    ax.plot(v1.k, v1.auc, "o", ms=20, mfc="none", mec=VERMILION, mew=2.5, zorder=5)
    ax.annotate("V1 (사전 지정 칸)", (v1.k, v1.auc), xytext=(6, 0.64), textcoords="data",
                color=VERMILION, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": VERMILION, "lw": 1.5})
    chance_line(ax)
    log_k_axis(ax)
    ax.set_ylim(lower_limit(pd.concat([d.ci_low, d.auc])), 1.0)
    ax.set_xlabel("한 사람에게서 모은 출력 수 k (로그 축)")
    ax.set_ylabel("다른 날 연결 AUC")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(color=GRAY, alpha=0.2, label="띠: 95% 구간"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              handlelength=1.6, labelspacing=0.7)
    save(fig, "v3_fig6_yang2025_comparators.png")


def main() -> None:
    style()
    rungs = [("features_P0", BLUE), ("features_P2-RC_d8", GREEN),
             ("commands_P2-RC_d8_tau0.80", VERMILION)]
    linkage_over_k("lee2019", rungs, "v3_fig1_lee2019_linkage_k.png", (1200, 640))
    linkage_over_k("bnci2014_001", rungs, "v3_fig5_bnci2014_001_linkage_k.png", (1240, 700))
    weak_attacker((1160, 500))
    yang2025_ladder((1100, 560))
    ladders_lee2019_2a((1728, 600))
    validation((1728, 560))
    yang2025_comparators((1240, 680))


if __name__ == "__main__":
    main()
