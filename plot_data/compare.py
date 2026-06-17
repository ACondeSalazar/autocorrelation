"""
compare.py — Macro comparison plot for all autocorrelation3D_* runs.

Usage
-----
python3 compare.py /path/to/root_dir [options]

The script scans root_dir for folders matching autocorrelation3D_* (or a
custom glob), loads each autocorrelation_raw.h5 inside, fits XY and Z
autocorrelation curves, then assembles a single summary figure per channel.

Layout (one figure per channel that has data):
  rows = runs  (sorted by the numeric suffix)
  cols = [XY autocorrelation | Z autocorrelation]

A summary CSV with fit parameters for all runs × channels is also written.

Example
-------
python3 compare.py /home/arthur/Documents/slices/elastix_transfer_full \
        --units physical --z-scale 2.18 --model triple --n-starts 100
"""

import argparse
import csv
import fnmatch
import os
import sys
import warnings

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from fit import fit_autocorrelation  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scale_label(xy_scale, z_scale, use_pixel_units, direction="xy"):
    """Return a clear scale annotation string."""
    if use_pixel_units:
        return "units: pixels (no physical scale applied)"
    if direction == "xy":
        return f"xy scale: {xy_scale:.4g} µm/px"
    else:
        return f"z scale: {z_scale:.4g} µm/slice"


def _numeric_suffix(folder_name: str) -> int:
    parts = folder_name.rstrip("/").split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def discover_runs(root: str, pattern: str = "autocorrelation3D_*") -> list[dict]:
    runs = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        if not fnmatch.fnmatch(entry.name, pattern):
            continue
        h5_path = os.path.join(entry.path, "autocorrelation_raw.h5")
        if not os.path.isfile(h5_path):
            h5_candidates = [f for f in os.listdir(entry.path) if f.endswith(".h5")]
            if not h5_candidates:
                print(f"  [skip] {entry.name}: no .h5 file found")
                continue
            h5_path = os.path.join(entry.path, h5_candidates[0])

        runs.append(
            {
                "label": entry.name,
                "folder": entry.path,
                "h5": h5_path,
                "sort_key": _numeric_suffix(entry.name),
            }
        )

    runs.sort(key=lambda r: r["sort_key"])
    return runs


def _load_curves(h5_path: str, ch_idx: int):
    with h5py.File(h5_path, "r") as f:
        xy_scale = float(f.attrs.get("xy_scale", 1.0))
        z_scale = float(f.attrs.get("z_scale", 1.0))

        meta = {
            "tile_size": f.attrs.get("tile_size", None),
            "z_stride": f.attrs.get("z_stride", None),
            "tile_depth_slices": f.attrs.get("tile_depth_slices", None),
            "avg_depth_slices": None,
            "n_tiles_xy": None,
            "n_tiles_xy_valid": None,
            "n_tiles_z": None,
            "n_tiles_z_valid": None,
        }

        if "fused_tiles" in f:
            fused_tiles = f["fused_tiles"][:]
            depths = fused_tiles[:, 3] - fused_tiles[:, 2]
            meta["avg_depth_slices"] = float(np.mean(depths))

        ch_key = f"channel_{ch_idx}"
        if ch_key not in f:
            return None, None, xy_scale, z_scale, meta

        ch_group = f[ch_key]
        courbe_xy = courbe_z = None

        if "radial_2d" in ch_group:
            radial_2d = ch_group["radial_2d"][:]
            valid = np.isfinite(radial_2d).any(axis=(1, 2))
            meta["n_tiles_xy"] = int(radial_2d.shape[0])
            meta["n_tiles_xy_valid"] = int(valid.sum())
            if valid.any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_tiles = np.nanmean(radial_2d[valid], axis=0)
                    courbe_xy = np.nanmean(mean_tiles, axis=0)

        if "profil_z" in ch_group:
            profil_z = ch_group["profil_z"][:]
            valid = np.isfinite(profil_z).any(axis=1)
            meta["n_tiles_z"] = int(profil_z.shape[0])
            meta["n_tiles_z_valid"] = int(valid.sum())
            if valid.any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    courbe_z = np.nanmean(profil_z[valid], axis=0)

    return courbe_xy, courbe_z, xy_scale, z_scale, meta


def _fit_xy(courbe_xy, xy_scale, use_pixel_units, model, n_starts):
    if courbe_xy is None or len(courbe_xy) < 3:
        return None, None, None
    r_vals = np.arange(1, len(courbe_xy))
    y_vals = courbe_xy[1:]
    if y_vals[0] <= 0 or not np.any(np.isfinite(y_vals)):
        return None, None, None
    y_vals = y_vals / y_vals[0]
    scale = 1.0 if use_pixel_units else xy_scale
    x_vals = r_vals * scale
    fit = fit_autocorrelation(r_vals, y_vals, model=model, n_starts=n_starts)
    return fit, x_vals, y_vals


def _fit_z(courbe_z, z_scale, use_pixel_units, model, n_starts):
    if courbe_z is None or len(courbe_z) < 3:
        return None, None, None
    max_val = np.nanmax(courbe_z)
    if not np.isfinite(max_val) or max_val <= 0:
        return None, None, None
    y_vals = courbe_z / max_val
    z_idx = np.arange(len(y_vals)).astype(float)
    scale = 1.0 if use_pixel_units else z_scale
    x_vals = z_idx * scale
    fit = fit_autocorrelation(z_idx, y_vals, model=model, n_starts=n_starts)
    return fit, x_vals, y_vals


_CMAP = plt.get_cmap("tab10")


def _run_color(run_idx: int):
    return _CMAP(run_idx % 10)


def make_comparison_figure(
    channel: int,
    runs: list[dict],
    results: dict,
    use_pixel_units: bool,
    output_path: str,
    xy_scale: float = 1.0,
    z_scale: float = 1.0,
    show_errors: bool = True,
):
    n_runs = len(runs)
    unit = "px" if use_pixel_units else "µm"

    has_xy = any(results[r["label"]]["xy"][0] is not None for r in runs)
    has_z = any(results[r["label"]]["z"][0] is not None for r in runs)

    n_cols = int(has_xy) + int(has_z)
    if n_cols == 0:
        print(f"  ch {channel}: nothing to plot, skipping figure")
        return

    INFO_H = 0.55 * n_runs
    PLOTS_H = 3.5 * n_runs
    fig_w = 8 * n_cols
    fig_h = INFO_H + PLOTS_H

    fig = plt.figure(figsize=(fig_w, fig_h))

    outer_rows = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[INFO_H, PLOTS_H],
        hspace=0.05,
    )

    # ── info strip ──────────────────────────────────────────────────────────
    ax_info = fig.add_subplot(outer_rows[0])
    ax_info.axis("off")
    ax_info.set_title(
        f"Autocorrelation comparison — channel {channel}",
        fontsize=13,
        fontweight="bold",
        pad=6,
        loc="left",
    )

    col_labels = [
        "Run",
        "tile_size (px)",
        "avg depth (sl)",
        "XY tiles (valid/total)",
        "Z tiles (valid/total)",
        "xy scale (µm/px)",
        "z scale (µm/slice)",
    ]
    table_data = []
    for run in runs:
        m = results[run["label"]].get("meta", {})
        ts = str(m.get("tile_size", "—"))
        dep = (
            f"{m['avg_depth_slices']:.1f}"
            if m.get("avg_depth_slices") is not None
            else str(m.get("tile_depth_slices", "—"))
        )
        xy_t = (
            f"{m['n_tiles_xy_valid']}/{m['n_tiles_xy']}"
            if m.get("n_tiles_xy") is not None
            else "—"
        )
        z_t = (
            f"{m['n_tiles_z_valid']}/{m['n_tiles_z']}"
            if m.get("n_tiles_z") is not None
            else "—"
        )
        xys = f"{xy_scale:.4g}" if not use_pixel_units else "—"
        zs = f"{z_scale:.4g}" if not use_pixel_units else "—"
        table_data.append([run["label"], ts, dep, xy_t, z_t, xys, zs])

    tbl = ax_info.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor("#d0d8e8")
        tbl[0, col_idx].set_text_props(fontweight="bold")
    for row_idx in range(1, len(table_data) + 1):
        shade = "#f5f5f5" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(col_labels)):
            tbl[row_idx, col_idx].set_facecolor(shade)

    # ── plots area ──────────────────────────────────────────────────────────
    outer = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=outer_rows[1], wspace=0.30
    )

    col_iter = iter(range(n_cols))

    for direction, has in [("xy", has_xy), ("z", has_z)]:
        if not has:
            continue
        col_idx = next(col_iter)

        inner = gridspec.GridSpecFromSubplotSpec(
            n_runs, 1, subplot_spec=outer[col_idx], hspace=0.55
        )

        xlabel = f"Distance XY ({unit})" if direction == "xy" else f"Depth Z ({unit})"
        dir_label = "XY" if direction == "xy" else "Z"
        scale_text = _scale_label(xy_scale, z_scale, use_pixel_units, direction)

        for row_idx, run in enumerate(runs):
            label = run["label"]
            color = _run_color(row_idx)

            fit, x_vals, y_vals = results[label][direction]

            # Each row: data+fit panel (top) + residuals panel (bottom)
            # When show_errors=False the residuals panel is hidden.
            if show_errors and fit is not None:
                row_gs = gridspec.GridSpecFromSubplotSpec(
                    2,
                    1,
                    subplot_spec=inner[row_idx],
                    height_ratios=[3, 1],
                    hspace=0.05,
                )
                ax_top = fig.add_subplot(row_gs[0])
                ax_bot = fig.add_subplot(row_gs[1], sharex=ax_top)
            else:
                row_gs = gridspec.GridSpecFromSubplotSpec(
                    1,
                    1,
                    subplot_spec=inner[row_idx],
                )
                ax_top = fig.add_subplot(row_gs[0])
                ax_bot = None

            if x_vals is not None and y_vals is not None:
                ax_top.plot(
                    x_vals, y_vals, "-", color=color, lw=1, alpha=0.8, label="data"
                )

            if fit is not None:
                scale_fit = (
                    xy_scale
                    if (direction == "xy" and not use_pixel_units)
                    else (
                        z_scale if (direction == "z" and not use_pixel_units) else 1.0
                    )
                )
                ax_top.plot(
                    x_vals,
                    fit.y_pred,
                    "--",
                    color="black",
                    lw=1.5,
                    label=f"Fit ({fit.model})  R²={fit.r2:.4f}  RMSE={fit.rmse:.4f}",
                )
                for pname, pval, perr in zip(fit.param_names, fit.popt, fit.perr):
                    if pname.startswith("l"):
                        disp = (
                            f"{pval * scale_fit:.3f}±{perr * scale_fit:.3f}{unit}"
                            if show_errors
                            else f"{pval * scale_fit:.3f}{unit}"
                        )
                    else:
                        disp = (
                            f"{pval:.3f}±{perr:.3f}" if show_errors else f"{pval:.3f}"
                        )
                    if show_errors and perr / max(abs(pval), 1e-12) > 0.5:
                        disp += "  ⚠"
                    ax_top.plot([], [], " ", label=f"  {pname} = {disp}")

                if show_errors and ax_bot is not None:
                    residuals = y_vals - fit.y_pred
                    ax_bot.plot(x_vals, residuals, "-", color=color, lw=0.8)
                    ax_bot.fill_between(x_vals, residuals, alpha=0.2, color=color)
                    ax_bot.axhline(0, color="k", lw=0.5)
            else:
                if ax_bot is not None:
                    ax_bot.set_visible(False)
                ax_top.text(
                    0.5,
                    0.5,
                    "fit failed",
                    transform=ax_top.transAxes,
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=9,
                )

            ax_top.set_title(f"{label}  [{dir_label}]", fontsize=8, pad=3, loc="left")
            ax_top.set_ylim(-0.25, 1.15)
            ax_top.axhline(0, color="k", lw=0.4, ls="--")
            ax_top.set_ylabel("Autocorr.", fontsize=7)
            ax_top.legend(fontsize=6, loc="upper right")
            ax_top.grid(True, ls="--", alpha=0.35)
            ax_top.tick_params(labelsize=7)

            # scale annotation and x-label placement
            if ax_bot is not None:
                plt.setp(ax_top.get_xticklabels(), visible=False)
                ax_bot.set_ylabel("Error Data/Fit", fontsize=7)
                ax_bot.set_xlabel(xlabel, fontsize=8)
                ax_bot.tick_params(labelsize=7)
                ax_bot.grid(True, ls="--", alpha=0.35)
                ax_bot.text(
                    0.98,
                    0.05,
                    scale_text,
                    transform=ax_bot.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                )
            else:
                ax_top.set_xlabel(xlabel, fontsize=8)
                ax_top.text(
                    0.98,
                    0.02,
                    scale_text,
                    transform=ax_top.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")




def make_overlay_figure(
    channel: int,
    runs: list[dict],
    results: dict,
    use_pixel_units: bool,
    output_path: str,
    xy_scale: float = 1.0,
    z_scale: float = 1.0,
    show_errors: bool = True,
):
    unit = "px" if use_pixel_units else "µm"
    has_xy = any(results[r["label"]]["xy"][0] is not None for r in runs)
    has_z = any(results[r["label"]]["z"][0] is not None for r in runs)
    n_cols = int(has_xy) + int(has_z)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f"Overlay — channel {channel}", fontsize=13)

    col_iter = iter(axes)

    for direction, has in [("xy", has_xy), ("z", has_z)]:
        if not has:
            continue
        ax = next(col_iter)
        xlabel = f"Distance XY ({unit})" if direction == "xy" else f"Depth Z ({unit})"
        scale_text = _scale_label(xy_scale, z_scale, use_pixel_units, direction)
        ax.set_title("XY autocorrelation" if direction == "xy" else "Z autocorrelation")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Autocorrelation (normalised)")
        ax.set_ylim(-0.25, 1.15)
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.grid(True, ls="--", alpha=0.35)

        for run_idx, run in enumerate(runs):
            label = run["label"]
            color = _run_color(run_idx)
            fit, x_vals, y_vals = results[label][direction]

            if x_vals is None:
                continue
            ax.plot(x_vals, y_vals, "-", color=color, lw=0.8, alpha=0.55)
            if fit is not None:
                ax.plot(
                    x_vals,
                    fit.y_pred,
                    "--",
                    color=color,
                    lw=1.8,
                    label=f"{label}  R²={fit.r2:.4f}",
                )

        ax.legend(fontsize=7, loc="upper right")
        ax.text(
            0.98,
            0.05,
            scale_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")




def write_csv(all_results: list[dict], output_path: str):
    rows = []
    for entry in all_results:
        base = {
            "run": entry["run"],
            "channel": entry["channel"],
            "direction": entry["direction"],
            "model": entry.get("model", ""),
            "R2": entry.get("r2", ""),
            "RMSE": entry.get("rmse", ""),
        }
        for name, val, err in zip(
            entry.get("param_names", []),
            entry.get("popt", []),
            entry.get("perr", []),
        ):
            base[f"{name}"] = f"{val:.6g}"
            base[f"{name}_err"] = f"{err:.6g}"
        rows.append(base)

    if not rows:
        return

    all_keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  CSV summary: {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Macro comparison plot across all autocorrelation3D_* runs."
    )
    parser.add_argument(
        "root_dir", help="Root directory containing autocorrelation3D_* sub-folders."
    )
    parser.add_argument("--pattern", default="autocorrelation3D_*")
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--units", choices=["pixel", "physical"], default="physical")
    parser.add_argument("--xy-scale", type=float, default=0.325)
    parser.add_argument("--z-scale", type=float, default=None)
    parser.add_argument(
        "--model",
        choices=[
            "single",
            "double",
            "triple",
            "gaussian",
            "damped_cosine",
            "stretched",
            "stretched_exp",
        ],
        default="triple",
    )
    parser.add_argument("--n-starts", type=int, default=50)
    parser.add_argument(
        "--channels", nargs="+", type=int, default=[0, 1, 2], metavar="CH"
    )
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="Hide residual subplots and ± uncertainty in legend entries.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    out_dir = (
        os.path.abspath(args.output_folder)
        if args.output_folder
        else os.path.join(root, "comparison")
    )
    os.makedirs(out_dir, exist_ok=True)

    use_pixel_units = args.units == "pixel"
    show_errors = not args.no_errors

    print(f"Scanning {root} for pattern '{args.pattern}' …")
    runs = discover_runs(root, pattern=args.pattern)
    if not runs:
        sys.exit(f"No runs found in {root} matching '{args.pattern}'.")
    print(f"Found {len(runs)} run(s):")
    for r in runs:
        print(f"  {r['label']}  →  {r['h5']}")

    channel_results = {ch: {} for ch in args.channels}
    all_csv_rows = []

    print("\nFitting …")
    for run in runs:
        label = run["label"]
        h5 = run["h5"]
        print(f"\n  [{label}]")

        for ch in args.channels:
            channel_results[ch][label] = {
                "xy": (None, None, None),
                "z": (None, None, None),
                "meta": {},
            }

            courbe_xy, courbe_z, xy_scale_h5, z_scale_h5, meta = _load_curves(h5, ch)
            xy_scale = args.xy_scale if args.xy_scale is not None else xy_scale_h5
            z_scale = args.z_scale if args.z_scale is not None else z_scale_h5

            channel_results[ch][label]["meta"] = meta

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_xy, x_xy, y_xy = _fit_xy(
                    courbe_xy, xy_scale, use_pixel_units, args.model, args.n_starts
                )
            channel_results[ch][label]["xy"] = (fit_xy, x_xy, y_xy)
            if fit_xy is not None:
                unit = "px" if use_pixel_units else "µm"
                print(
                    f"    ch{ch} XY | {fit_xy.summary(scale=xy_scale if not use_pixel_units else 1.0, unit=unit)}"
                )
                all_csv_rows.append(
                    {
                        "run": label,
                        "channel": ch,
                        "direction": "xy",
                        "model": fit_xy.model,
                        "r2": fit_xy.r2,
                        "rmse": fit_xy.rmse,
                        "param_names": fit_xy.param_names,
                        "popt": fit_xy.popt,
                        "perr": fit_xy.perr,
                    }
                )
            else:
                print(f"    ch{ch} XY | fit failed / no data")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_z, x_z, y_z = _fit_z(
                    courbe_z, z_scale, use_pixel_units, args.model, args.n_starts
                )
            channel_results[ch][label]["z"] = (fit_z, x_z, y_z)
            if fit_z is not None:
                unit = "px" if use_pixel_units else "µm"
                print(
                    f"    ch{ch} Z  | {fit_z.summary(scale=z_scale if not use_pixel_units else 1.0, unit=unit)}"
                )
                all_csv_rows.append(
                    {
                        "run": label,
                        "channel": ch,
                        "direction": "z",
                        "model": fit_z.model,
                        "r2": fit_z.r2,
                        "rmse": fit_z.rmse,
                        "param_names": fit_z.param_names,
                        "popt": fit_z.popt,
                        "perr": fit_z.perr,
                    }
                )
            else:
                print(f"    ch{ch} Z  | fit failed / no data")

    print("\nBuilding figures …")
    for ch in args.channels:
        stacked_path = os.path.join(out_dir, f"comparison_ch{ch}_stacked.png")
        make_comparison_figure(
            channel=ch,
            runs=runs,
            results=channel_results[ch],
            use_pixel_units=use_pixel_units,
            output_path=stacked_path,
            xy_scale=args.xy_scale if args.xy_scale is not None else 1.0,
            z_scale=args.z_scale if args.z_scale is not None else 1.0,
            show_errors=show_errors,
        )

        if not args.no_overlay:
            overlay_path = os.path.join(out_dir, f"comparison_ch{ch}_overlay.png")
            make_overlay_figure(
                channel=ch,
                runs=runs,
                results=channel_results[ch],
                use_pixel_units=use_pixel_units,
                output_path=overlay_path,
                xy_scale=args.xy_scale if args.xy_scale is not None else 1.0,
                z_scale=args.z_scale if args.z_scale is not None else 1.0,
                show_errors=show_errors,
            )

    csv_path = os.path.join(out_dir, "comparison_summary.csv")
    write_csv(all_csv_rows, csv_path)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
