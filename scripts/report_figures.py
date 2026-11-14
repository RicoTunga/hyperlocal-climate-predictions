"""Chapter 5 report figures, made only from the committed result files in reports/.

Run from the repo root (venv active):

    python scripts/report_figures.py                 # writes docs/report/latex/figures/ch5/*.pdf
    python scripts/report_figures.py --preview DIR   # also writes PNG previews to DIR

The script only READS result files; it never writes to reports/. Each figure is drawn at the
A4 text width of the report (A4 with 1-inch margins: 210 mm - 2 x 25.4 mm = 6.27 in), so the
PDF is included at width=\\textwidth without scaling, and every text element is checked to be
at least 9 pt before saving.

Figures (numbers follow docs/report/ch5_outline.md):
  fig5_3_validation_algorithms.pdf  validation 2024 skill by lead time, three algorithms
  fig5_5_shap_groups.pdf            SHAP share by input group, days 1/3/7, overall and within-day
  fig5_6_test_skill.pdf             test skill by lead time vs climatology and vs persistence
  fig5_7_test_calibration.pdf       test calibration: bias (95% CI) and slope by lead time
  fig5_8_test_periods.pdf           test day-1 skill by period and season
Figures 5.1, 5.2 and 5.4 in the outline are a map, a data sample and a workflow diagram; they
are not drawn from result files and are not made here.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "docs" / "report" / "latex" / "figures" / "ch5"

TEXT_WIDTH_IN = (210 - 2 * 25.4) / 25.4   # 6.27 in
MIN_PT = 9

# Colours: reference palette (dataviz skill), validated for these uses on a white page:
# categorical slots 1-3 (all pairs pass CVD/normal-vision checks; aqua is < 3:1 contrast, so every
# series also has its own marker shape and a legend) and a blue ordinal ramp for days ahead.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {1: "#86b6ef", 3: "#2a78d6", 7: "#104281"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SITES = {"nakuru": "Nakuru", "busia": "Busia"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": MIN_PT, "axes.titlesize": MIN_PT,
        "axes.labelsize": MIN_PT, "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT,
        "legend.fontsize": MIN_PT, "axes.titleweight": "bold",
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def read(rel: str) -> pd.DataFrame:
    """Read one result file (read-only)."""
    return pd.read_csv(REPORTS / rel)


def one(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Rows matching every key=value; a value of None matches a missing (NaN) cell."""
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= df[k].isna().to_numpy() if v is None else (df[k] == v).to_numpy()
    return df[m]


def points(ax, x, est, lo, hi, color, marker, label, dx=0.0):
    """Dots with 95% interval bars (no line between lead times: each day is its own estimate)."""
    x = np.asarray(x, float) + dx
    est, lo, hi = (np.asarray(a, float) for a in (est, lo, hi))
    ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt=marker, color=color, ms=5, mec="white",
                mew=0.6, elinewidth=1.4, capsize=0, label=label, zorder=3)


def zero_line(ax, y=0.0):
    ax.axhline(y, color=INK2, lw=0.9, zorder=2)


def days_axis(ax):
    ax.set_xticks(range(1, 8))
    ax.set_xlim(0.5, 7.5)
    ax.grid(axis="x", visible=False)


def label_grid(axes, row_labels):
    """Site names as column titles (top row), measure as the row's y label, shared y range per row."""
    for c, site in enumerate(SITES):
        axes[0, c].set_title(SITES[site], loc="left")
    for r, lab in enumerate(row_labels):
        axes[r, 0].set_ylabel(lab)
        lo = min(ax.get_ylim()[0] for ax in axes[r])
        hi = max(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:
            ax.set_ylim(lo, hi)
    for ax in axes[-1]:
        ax.set_xlabel("Days ahead")


def legend_below(fig, title=None, ncols=3):
    """One legend for the whole figure, below the panels, so it never covers data."""
    handles, labels = fig.axes[0].get_legend_handles_labels()
    kw = dict(loc="outside lower center", ncols=ncols, handletextpad=0.3, columnspacing=1.5)
    if title:
        kw.update(title=title, title_fontsize=MIN_PT)
    fig.legend(handles, labels, **kw)


def check_fonts(fig) -> None:
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    small = [s for s in sizes if s < MIN_PT]
    if small:
        raise ValueError(f"{len(small)} text elements below {MIN_PT} pt: {sorted(set(small))}")


def save(fig, name: str, preview: Path | None, made: list) -> None:
    w, h = fig.get_size_inches()
    assert abs(w - TEXT_WIDTH_IN) < 1e-6, w
    check_fonts(fig)
    path = OUT / f"{name}.pdf"
    fig.savefig(path)                      # no bbox "tight": keeps the exact text-width size
    if preview:
        fig.savefig(preview / f"{name}.png", dpi=150)
    plt.close(fig)
    made.append((path, w, h))


# ------------------------------------------------------------------ figures
SKILL = [("rain>=1mm", "bss", "Rain ≥ 1 mm\nBrier skill"),
         ("rh_pct", "rmse_skill", "Relative humidity\nRMSE skill")]


def fig_validation_algorithms(preview, made):
    """Figure 5.3: validation 2024, 60 training / 60 scored pixels, skill vs climatology."""
    v = read("skill_ci_validation.csv")
    v = v[(v.n_train_pixels == 60) & (v.n_score_pixels == 60)]
    algos = [("catboost", "CatBoost", BLUE, "o", -0.2), ("lightgbm", "LightGBM", ORANGE, "s", 0.0),
             ("xgboost", "XGBoost", AQUA, "^", 0.2)]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 5.2), layout="constrained", sharex=True)
    for r, (target, metric, label) in enumerate(SKILL):
        for c, site in enumerate(SITES):
            ax = axes[r, c]
            for algo, name, col, mk, dx in algos:
                d = one(v, site=site, model=algo, target=target, metric=metric,
                        reference="climatology").sort_values("horizon")
                assert len(d) == 7, (site, algo, target)
                points(ax, d.horizon, d.estimate, d.ci_low, d.ci_high, col, mk, name, dx)
            zero_line(ax)
            days_axis(ax)
    label_grid(axes, [f"{lab}\nvs climatology" for _, _, lab in SKILL])
    legend_below(fig)
    save(fig, "fig5_3_validation_algorithms", preview, made)


def fig_test_skill(preview, made):
    """Figure 5.6: final test, CatBoost, full period, skill vs climatology and vs persistence."""
    s = read("test/skill.csv")
    s = one(s, period="full test period", model="catboost")
    refs = [("climatology", "vs climatology", BLUE, "o", -0.12),
            ("persistence", "vs persistence", ORANGE, "s", 0.12)]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 5.2), layout="constrained", sharex=True)
    for r, (target, metric, label) in enumerate(SKILL):
        for c, site in enumerate(SITES):
            ax = axes[r, c]
            for ref, name, col, mk, dx in refs:
                d = one(s, site=site, target=target, metric=metric, reference=ref).sort_values("horizon")
                assert len(d) == 7, (site, ref, target)
                points(ax, d.horizon, d.estimate, d.ci_low, d.ci_high, col, mk, name, dx)
            zero_line(ax)
            days_axis(ax)
    label_grid(axes, [lab for _, _, lab in SKILL])
    legend_below(fig, ncols=2)
    save(fig, "fig5_6_test_skill", preview, made)


def fig_test_calibration(preview, made):
    """Figure 5.7: final test, CatBoost: bias (forecast minus observed rate) with 95% CI, and slope."""
    cal = read("test/calibration.csv")
    cal = cal[cal.model == "catboost"]
    thr = [(1.0, "≥ 1 mm", BLUE, "o", -0.2), (10.0, "≥ 10 mm", ORANGE, "s", 0.0),
           (25.0, "≥ 25 mm", AQUA, "^", 0.2)]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 5.2), layout="constrained", sharex=True)
    for c, site in enumerate(SITES):
        for t, name, col, mk, dx in thr:
            d = one(cal, site=site, threshold_mm=t).sort_values("horizon")
            assert len(d) == 7, (site, t)
            points(axes[0, c], d.horizon, d.bias, d.bias_ci_low, d.bias_ci_high, col, mk, name, dx)
            axes[1, c].plot(d.horizon + dx, d.slope, mk, color=col, ms=5, mec="white", mew=0.6,
                            label=name, zorder=3)
        zero_line(axes[0, c], 0.0)
        zero_line(axes[1, c], 1.0)
        for ax in axes[:, c]:
            days_axis(ax)
    label_grid(axes, ["Bias: mean forecast\nminus observed rate", "Calibration slope\n(1 = calibrated)"])
    legend_below(fig, title="Rain threshold")
    save(fig, "fig5_7_test_calibration", preview, made)


def fig_test_periods(preview, made):
    """Figure 5.8: final test, CatBoost, day 1, skill vs climatology by period and season."""
    s = read("test/skill.csv")
    s = one(s, model="catboost", horizon=1, reference="climatology")
    periods = ["full test period", "2025", "2026 (Jan-Sep)", "season JF", "season MAM",
               "season JJAS", "season OND"]
    labels = ["Full period", "2025", "2026 (Jan–Sep)", "Jan–Feb", "Mar–May", "Jun–Sep", "Oct–Dec"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.8), layout="constrained", sharey=True)
    y = np.arange(len(periods))[::-1]
    titles = ["Rain ≥ 1 mm (Brier skill)", "Relative humidity (RMSE skill)"]
    for ax, (target, metric, _), label in zip(axes, SKILL, titles):
        for (site, name), col, mk, dy in [(("nakuru", "Nakuru"), BLUE, "o", 0.15),
                                          (("busia", "Busia"), ORANGE, "s", -0.15)]:
            d = one(s, site=site, target=target, metric=metric).set_index("period").loc[periods]
            ax.errorbar(d.estimate, y + dy, xerr=[d.estimate - d.ci_low, d.ci_high - d.estimate],
                        fmt=mk, color=col, ms=5, mec="white", mew=0.6, elinewidth=1.4, capsize=0,
                        label=name, zorder=3)
        ax.axvline(0, color=INK2, lw=0.9, zorder=2)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Day-1 skill vs climatology")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    axes[0].set_yticks(y, labels)
    legend_below(fig, ncols=2)
    save(fig, "fig5_8_test_periods", preview, made)


def fig_shap_groups(preview, made):
    """Figure 5.5: share of mean |SHAP| by input group (validation 2024, CatBoost)."""
    g = read("shap_groups_validation.csv")
    order = ["area-wide", "own ERA5 weather", "own rain history", "neighbourhood rain",
             "climatology", "calendar", "location & elevation"]
    names = ["Area-wide", "Own ERA5 weather", "Own rain history", "Neighbourhood rain",
             "Climatology", "Calendar", "Location & elevation"]
    assert set(g.group) == set(order), set(g.group)
    tasks = [("occ1", "Rain ≥ 1 mm"), ("wet", "Wet-day amount"), ("rh", "Humidity")]
    cols = [("nakuru", "overall_share", "Nakuru\noverall"),
            ("nakuru", "within_day_share", "Nakuru\nwithin day"),
            ("busia", "overall_share", "Busia\noverall"),
            ("busia", "within_day_share", "Busia\nwithin day")]
    fig, axes = plt.subplots(3, 4, figsize=(TEXT_WIDTH_IN, 6.6), layout="constrained",
                             sharex=True, sharey=True)
    y = np.arange(len(order))[::-1]
    for r, (task, tname) in enumerate(tasks):
        for c, (site, col, cname) in enumerate(cols):
            ax = axes[r, c]
            for h, dy in [(1, 0.22), (3, 0.0), (7, -0.22)]:
                d = one(g, site=site, task=task, horizon=h).set_index("group").loc[order]
                assert abs(d[col].sum() - 1) < 1e-9, (site, task, h, col)
                ax.plot(d[col], y + dy, "o", color=RAMP[h], ms=4.5, mec="white", mew=0.5, zorder=3)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, 0.85)
            ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", ".25", ".5", ".75"])
            if r == 0:
                ax.set_title(cname, loc="left")
            if c == 0:
                ax.set_ylabel(tname, fontweight="bold", color=INK)
            if r == 2:
                ax.set_xlabel("Share")
    axes[0, 0].set_yticks(y, names)
    handles = [Line2D([], [], ls="", marker="o", color=RAMP[h], ms=5, label=f"Day {h}") for h in (1, 3, 7)]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, title="Days ahead",
               title_fontsize=MIN_PT)
    save(fig, "fig5_5_shap_groups", preview, made)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, help="also write PNG previews to this folder")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
    style()
    made = []
    for f in (fig_validation_algorithms, fig_shap_groups, fig_test_skill, fig_test_calibration,
              fig_test_periods):
        f(args.preview, made)
    print(f"Wrote {len(made)} figures to {OUT.relative_to(ROOT)} (text width {TEXT_WIDTH_IN:.2f} in, "
          f"all text >= {MIN_PT} pt):")
    for path, w, h in made:
        print(f"  {path.name:34s} {w:.2f} x {h:.2f} in ({w * 25.4:.0f} x {h * 25.4:.0f} mm)")


if __name__ == "__main__":
    main()
