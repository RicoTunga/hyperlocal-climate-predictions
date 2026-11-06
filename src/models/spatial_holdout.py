"""Spatial hold-out: does the model work on pixels far from any pixel it was trained on?

Run from the repo root after the decided training run (train60, scored on all pixels):

    python -m src.models.spatial_holdout

Design (settings in config/experiment.yaml, section spatial_holdout)
- Each site's grid is split into 2 x 2 blocks at the middle grid row and column (SW, SE, NW, NE).
  Every pixel belongs to exactly one block, so every pixel is held out exactly once.
- For each block: the training pool is every pixel MORE than `buffer_pixels` grid cells
  (Chebyshev distance) from the block. 60 training pixels (modelling.dev_pixels) are drawn from the
  pool, evenly spaced by elevation rank, as in the decided setup. CatBoost (default settings) is
  trained exactly as in src.models.train and scored on the held-out block, validation 2024.
- Features are used as built: CHIRPS covers every pixel, so a location the model was not trained
  on still has its own rain history and its area-wide and neighbourhood context.
- Headline comparison: the four held-out blocks together give, for every pixel, a forecast from a
  model that never trained within buffer_pixels of it. This is compared, paired on identical rows,
  with the standard model (trained on 60 pixels spread over the whole site) on the same pixels and
  days, using the moving-block bootstrap of src.models.uncertainty. Per-block comparisons as well.

Output: models/<site>/holdout_<block>/h<h>/, reports/spatial_holdout_skill_validation.csv,
reports/spatial_holdout_differences_validation.csv, reports/figures/spatial_holdout_design.png.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import grid_index
from src.models.baselines import fit_climatology
from src.models.calibration import decided_run
from src.models.evaluation import load_configs, split_bounds
from src.models.train import (MODELS, load_features, run_site_horizon, site_pixels, spread_by_elevation)
from src.models.uncertainty import (block_bootstrap_weights, bootstrap_site_horizon, combine_runs,
                                    paired_differences)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
BLOCKS = ["SW", "SE", "NW", "NE"]
ALGO = "catboost"


def assign_blocks(z: pd.DataFrame) -> pd.DataFrame:
    """Add grid indices (gi = latitude row, gj = longitude column) and a 2 x 2 block name."""
    z = z.copy()
    z["gi"], z["gj"] = grid_index(z["lat"].to_numpy()), grid_index(z["lon"].to_numpy())
    north = z["gi"] >= (z["gi"].max() + 1) / 2
    east = z["gj"] >= (z["gj"].max() + 1) / 2
    z["block"] = np.select([~north & ~east, ~north & east, north & ~east], ["SW", "SE", "NW"], "NE")
    return z


def training_pool(z: pd.DataFrame, block: str, buffer: int) -> pd.DataFrame:
    """Pixels more than `buffer` grid cells (Chebyshev distance) from every pixel of `block`."""
    held = z[z["block"] == block]
    d = np.max(np.abs(z[["gi", "gj"]].to_numpy()[:, None, :] - held[["gi", "gj"]].to_numpy()[None, :, :]), axis=2)
    return z[d.min(axis=1) > buffer]


def design_figure(designs: dict, out: Path) -> None:
    """Map of each fold: training pixels, buffer/unused pixels, held-out block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "surface": "#fcfcfb",
           "unused": "#e1e0d9"}
    colors = {"held out": "#eb6834", "training": "#2a78d6", "not used": ink["unused"]}
    sites = list(designs)
    fig, axes = plt.subplots(len(sites), len(BLOCKS), figsize=(3.0 * len(BLOCKS), 3.2 * len(sites)),
                             dpi=150, facecolor=ink["surface"], squeeze=False)
    for i, site in enumerate(sites):
        for j, block in enumerate(BLOCKS):
            ax = axes[i, j]
            z = designs[site][block]
            for role in ["not used", "training", "held out"]:
                s = z[z["role"] == role]
                ax.scatter(s["lon"], s["lat"], s=9 if site == "nakuru" else 22, marker="s", color=colors[role],
                           linewidths=0, label=role)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            n_tr, n_ho = (z["role"] == "training").sum(), (z["role"] == "held out").sum()
            ax.set_title(f"{site.capitalize()} {block}: train {n_tr}, held out {n_ho}", fontsize=8.5,
                         color=ink["primary"], loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=ink["secondary"], markerscale=1.8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Spatial hold-out design: each 2 x 2 block held out once; grey = buffer and unsampled pixels",
                 fontsize=9.5, color=ink["primary"], y=1.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=ink["surface"])
    plt.close(fig)


def main() -> None:
    sites_cfg, exp = load_configs()
    scfg, mcfg = exp["spatial_holdout"], exp["modelling"]
    wet_mm, exceed, window = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"], exp["climatology"]["window_days"]
    ucfg = exp["uncertainty"]
    max_h = max(sites_cfg["forecast"]["horizons"])
    train_start, train_end = split_bounds(exp, "train")
    val_start, val_end = split_bounds(exp, "validation")

    skill_rows, diff_rows, designs = [], [], {}
    for site in sites_cfg["sites"]:
        z = assign_blocks(site_pixels(site))
        standard = dict(decided_run(site, mcfg["dev_pixels"]))  # horizon -> standard predictions file
        designs[site] = {}
        holdout_preds = {h: [] for h in scfg["horizons"]}
        for block in BLOCKS:
            held = sorted(z.loc[z["block"] == block, "pixel_id"])
            pool = training_pool(z, block, scfg["buffer_pixels"])
            train_px = spread_by_elevation(pool, mcfg["dev_pixels"])
            assert not set(train_px) & set(held)
            role = np.where(z["pixel_id"].isin(held), "held out",
                            np.where(z["pixel_id"].isin(train_px), "training", "not used"))
            designs[site][block] = z.assign(role=role)
            print(f"\n=== {site} {block}: hold out {len(held)} pixels, pool {len(pool)} pixels beyond "
                  f"{scfg['buffer_pixels']} cells, training on {len(train_px)} "
                  f"(elevation {pool.loc[pool['pixel_id'].isin(train_px), 'elev_mean_m'].min():.0f}-"
                  f"{pool.loc[pool['pixel_id'].isin(train_px), 'elev_mean_m'].max():.0f} m) ===")

            feat = load_features(site, train_px, until=val_end)
            clim = fit_climatology(feat[(feat["date"] >= train_start) & (feat["date"] <= train_end)],
                                   wet_mm, exceed, window)
            hist = load_features(site, held, until=train_end, start=train_start,
                                 columns=["pixel_id", "date", "precip_mm", "rh_pct"])
            score = (load_features(site, held, until=val_end, start=val_start),
                     fit_climatology(hist, wet_mm, exceed, window))
            del hist
            tag = f"holdout_{block}"
            for h in scfg["horizons"]:
                run_site_horizon(site, feat, clim, h, exp, [ALGO], max_h, score=score, out_tag=tag)
                holdout_preds[h].append(pd.read_parquet(MODELS / site / tag / f"h{h}" / "val_predictions.parquet"))
            del feat, score

        # paired comparison on identical rows: hold-out model vs standard model, per block and pooled
        for h in scfg["horizons"]:
            ho_all = pd.concat(holdout_preds[h], ignore_index=True)
            std = pd.read_parquet(standard[h])
            assert set(ho_all["pixel_id"]) == set(std["pixel_id"]), "blocks must cover every pixel exactly once"
            assert not ho_all.duplicated(["pixel_id", "target_date"]).any()
            for scope, ho in [("all pixels", ho_all)] + [(b, p) for b, p in zip(BLOCKS, holdout_preds[h])]:
                st = std[std["pixel_id"].isin(set(ho["pixel_id"]))]
                pr = combine_runs({"holdout": ho, "standard": st}, keep_models=[ALGO])
                W = block_bootstrap_weights(pr["target_date"].nunique(), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
                rows, draws = bootstrap_site_horizon(pr, W, wet_mm, exceed)
                for r in rows:
                    if r["reference"] == "climatology":
                        skill_rows.append({"site": site, "scope": scope, "horizon": h,
                                           "n_pixels": ho["pixel_id"].nunique(), **r})
                for d in paired_differences(draws, pd.DataFrame(rows)):
                    if "two-stage" in d["comparison"]:
                        continue
                    if d["comparison"].startswith(f"{ALGO}@standard"):  # report hold-out minus standard
                        d = {**d, "comparison": f"{ALGO}@holdout - {ALGO}@standard", "estimate": -d["estimate"],
                             "ci_low": -d["ci_high"], "ci_high": -d["ci_low"]}
                    diff_rows.append({"site": site, "scope": scope, "horizon": h,
                                      "n_pixels": ho["pixel_id"].nunique(), **d})

    skill, diff = pd.DataFrame(skill_rows), pd.DataFrame(diff_rows)
    skill.to_csv(REPORTS / "spatial_holdout_skill_validation.csv", index=False)
    diff.to_csv(REPORTS / "spatial_holdout_differences_validation.csv", index=False)
    design_figure(designs, FIGURES / "spatial_holdout_design.png")

    def fmt(r):
        star = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else " "
        return f"{r['estimate']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{star}"

    show = ["rain>=1mm", "rain>=10mm", "rain_mm_wet", "rh_pct"]
    for site in skill["site"].unique():
        s = skill[(skill["site"] == site) & (skill["scope"] == "all pixels") & skill["target"].isin(show)]
        print(f"\n{site}, all pixels: skill vs climatology (hold-out model / standard model)")
        for model in [f"{ALGO}@holdout", f"{ALGO}@standard"]:
            t = s[s["model"] == model].assign(v=lambda d: d.apply(fmt, axis=1)).pivot(index="horizon", columns="target", values="v")
            print(f"  {model}:\n    " + t[show].to_string().replace("\n", "\n    "))
        d = diff[(diff["site"] == site) & diff["target"].isin(show)]
        t = d[d["scope"] == "all pixels"].assign(v=lambda x: x.apply(fmt, axis=1)).pivot(index="horizon", columns="target", values="v")
        print("  hold-out minus standard:\n    " + t[show].to_string().replace("\n", "\n    "))
        blk = d[d["scope"] != "all pixels"]
        sig = blk[(blk["ci_low"] > 0) | (blk["ci_high"] < 0)]
        print(f"  per-block differences whose interval excludes 0: {len(sig)} of {len(blk)}"
              + (f" -> {sorted(zip(sig['scope'], sig['horizon'], sig['target'], sig['estimate'].round(3)))}" if len(sig) else ""))
    print(f"\nsaved {REPORTS / 'spatial_holdout_skill_validation.csv'}, "
          f"{REPORTS / 'spatial_holdout_differences_validation.csv'}, {FIGURES / 'spatial_holdout_design.png'}")


if __name__ == "__main__":
    main()
