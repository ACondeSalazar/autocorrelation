import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from fit import FitResult, fit_autocorrelation


def _scale_label(xy_scale, z_scale, use_pixel_units, direction="xy"):
    """Return a clear scale annotation string."""
    if use_pixel_units:
        return "units: pixels (no physical scale applied)"
    if direction == "xy":
        return f"xy scale: {xy_scale:.4g} µm/px"
    else:
        return f"z scale: {z_scale:.4g} µm/slice"


def plot_xy_autocorrelation(
    courbe_xy: np.ndarray,
    ch_idx: int,
    xy_scale: float,
    use_pixel_units: bool,
    output_folder: str,
    model: str = "triple",
    n_starts: int = 50,
    show_errors: bool = True,
) -> FitResult | None:
    r_vals = np.arange(1, len(courbe_xy))
    y_vals = courbe_xy[1:]
    if y_vals[0] <= 0 or not np.any(np.isfinite(y_vals)):
        print(f"ch {ch_idx} XY: invalid data, skipping")
        return None
    y_vals = y_vals / y_vals[0]

    scale = 1.0 if use_pixel_units else xy_scale
    unit = "px" if use_pixel_units else "µm"
    xlabel = f"Distance XY ({unit})"
    x_vals = r_vals * scale

    fit = fit_autocorrelation(r_vals, y_vals, model=model, n_starts=n_starts)

    if show_errors and fit is not None:
        fig = plt.figure(figsize=(10, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = None

    ax1.plot(x_vals, y_vals, "b-", lw=1, label="Data XY")

    if fit is not None:
        print(f"ch {ch_idx} XY | {fit.summary(scale=scale, unit=unit)}")
        if w := fit.warning():
            print(f"  ⚠ {w}")

        ax1.plot(
            x_vals,
            fit.y_pred,
            "g--",
            lw=2,
            label=f"Fit ({fit.model})  R²={fit.r2:.4f}  RMSE={fit.rmse:.4f}",
        )

        for name, val, err in zip(fit.param_names, fit.popt, fit.perr):
            if name.startswith("l"):
                disp = (
                    f"{val * scale:.3f}±{err * scale:.3f}{unit}"
                    if show_errors
                    else f"{val * scale:.3f}{unit}"
                )
            else:
                disp = f"{val:.3f}±{err:.3f}" if show_errors else f"{val:.3f}"
            ax1.plot([], [], " ", label=f"  {name} = {disp}")

        if show_errors and ax2 is not None:
            residuals = y_vals - fit.y_pred
            ax2.plot(x_vals, residuals, "r-", lw=0.8)
            ax2.fill_between(x_vals, residuals, alpha=0.2, color="red")
    else:
        print(f"ch {ch_idx} XY fit failed")

    ax1.axhline(0, color="k", lw=0.5, ls="--")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_ylim(-0.2, 1.1)
    ax1.legend(fontsize=8)
    ax1.set_title(f"Cylindrical autocorrelation (XY) - channel {ch_idx}")
    ax1.grid(True, ls="--", alpha=0.4)

    scale_text = _scale_label(xy_scale, None, use_pixel_units, direction="xy")
    if ax2 is not None:
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel("Error Fit/Points")
        ax2.set_xlabel(xlabel)
        ax2.grid(True, ls="--", alpha=0.4)
        ax2.text(
            0.98,
            0.05,
            scale_text,
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    else:
        ax1.set_xlabel(xlabel)
        ax1.text(
            0.98,
            0.02,
            scale_text,
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )

    path = os.path.join(output_folder, f"xy_autocorrelation_channel_{ch_idx}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fit


def plot_z_autocorrelation(
    courbe_z: np.ndarray,
    ch_idx: int,
    z_scale: float,
    use_pixel_units: bool,
    output_folder: str,
    model: str = "triple",
    n_starts: int = 50,
    show_errors: bool = True,
) -> FitResult | None:
    max_z_val = np.nanmax(courbe_z)
    if not np.isfinite(max_z_val) or max_z_val <= 0:
        print(f"ch {ch_idx} Z: invalid profile, skipping")
        return None
    courbe_z = courbe_z / max_z_val

    z_idx = np.arange(len(courbe_z))
    scale = 1.0 if use_pixel_units else z_scale
    unit = "px" if use_pixel_units else "µm"
    xlabel = f"Depth Z ({unit})"
    z_vals = z_idx * scale

    fit = fit_autocorrelation(
        z_idx.astype(float), courbe_z, model=model, n_starts=n_starts
    )

    if show_errors and fit is not None:
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = None

    ax1.plot(z_vals, courbe_z, color="red", lw=1, label="Data Z")

    if fit is not None:
        print(f"ch {ch_idx} Z  | {fit.summary(scale=scale, unit=unit)}")
        if w := fit.warning():
            print(f"  ⚠ {w}")

        ax1.plot(
            z_vals,
            fit.y_pred,
            "k--",
            lw=2,
            label=f"Fit ({fit.model})  R²={fit.r2:.4f}  RMSE={fit.rmse:.4f}",
        )

        for name, val, err in zip(fit.param_names, fit.popt, fit.perr):
            if name.startswith("l"):
                disp = (
                    f"{val * scale:.3f}±{err * scale:.3f}{unit}"
                    if show_errors
                    else f"{val * scale:.3f}{unit}"
                )
            else:
                disp = f"{val:.3f}±{err:.3f}" if show_errors else f"{val:.3f}"
            ax1.plot([], [], " ", label=f"  {name} = {disp}")

        if show_errors and ax2 is not None:
            residuals = courbe_z - fit.y_pred
            ax2.plot(z_vals, residuals, "r-", lw=0.8)
            ax2.fill_between(z_vals, residuals, alpha=0.2, color="red")
    else:
        print(f"ch {ch_idx} Z fit failed")

    ax1.axhline(0, color="k", lw=0.5, ls="--")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_ylim(-0.3, 1.1)
    ax1.legend(fontsize=8)
    ax1.set_title(f"Autocorrelation across Z - channel {ch_idx}")
    ax1.grid(True, ls="--", alpha=0.4)

    scale_text = _scale_label(None, z_scale, use_pixel_units, direction="z")
    if ax2 is not None:
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel("Error Fit/Points")
        ax2.set_xlabel(xlabel)
        ax2.grid(True, ls="--", alpha=0.4)
        ax2.text(
            0.98,
            0.05,
            scale_text,
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    else:
        ax1.set_xlabel(xlabel)
        ax1.text(
            0.98,
            0.02,
            scale_text,
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )

    path = os.path.join(output_folder, f"z_autocorrelation_channel_{ch_idx}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fit


def main():
    import argparse

    import h5py

    parser = argparse.ArgumentParser(
        description="Plot autocorrelation from saved HDF5 data"
    )
    parser.add_argument("h5_file")
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--units", choices=["pixel", "physical"], default="physical")
    parser.add_argument("--xy-scale", type=float, default=None)
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
        "--no-errors",
        action="store_true",
        help="Hide residual subplot and ± uncertainty in legend entries.",
    )
    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = os.path.dirname(os.path.abspath(args.h5_file))
    os.makedirs(args.output_folder, exist_ok=True)

    use_pixel_units = args.units == "pixel"
    show_errors = not args.no_errors

    with h5py.File(args.h5_file, "r") as f:
        xy_scale = float(f.attrs["xy_scale"])
        z_scale = float(f.attrs["z_scale"])
        if args.xy_scale is not None:
            xy_scale = args.xy_scale
        if args.z_scale is not None:
            z_scale = args.z_scale

        if "fused_tiles" in f:
            fused_tiles = f["fused_tiles"][:]
            depths = fused_tiles[:, 3] - fused_tiles[:, 2]
            unique_depths = sorted(set(depths))
            max_z_len = max(depths)
            global_z_center = max_z_len // 2
            global_max_dz = min(global_z_center, max_z_len - 1 - global_z_center)
            print(f"Fused tile depth groups: {unique_depths}")
            print(f"max_dz: {global_max_dz}")
        for attr in ("tile_size", "z_stride", "tile_depth_slices"):
            if attr in f.attrs:
                print(f"{attr}: {f.attrs[attr]}")

        for ch_idx in [0, 1, 2]:
            ch_key = f"channel_{ch_idx}"
            if ch_key not in f:
                continue

            ch_group = f[ch_key]
            radial_2d = ch_group["radial_2d"][:]
            profil_z = ch_group["profil_z"][:]

            valid_xy = np.isfinite(radial_2d).any(axis=(1, 2))
            valid_z = np.isfinite(profil_z).any(axis=1)
            total = radial_2d.shape[0]
            print(
                f"ch {ch_idx} - tiles used: XY {valid_xy.sum()}/{total}, "
                f"Z {valid_z.sum()}/{total}"
            )

            if not valid_xy.any() and not valid_z.any():
                print(f"ch {ch_idx}: no valid tiles, skipping")
                continue

            if valid_xy.any():
                try:
                    mean_tiles_xy = np.nanmean(radial_2d[valid_xy], axis=0)
                    courbe_xy = np.nanmean(mean_tiles_xy, axis=0)
                    plot_xy_autocorrelation(
                        courbe_xy,
                        ch_idx,
                        xy_scale,
                        use_pixel_units,
                        args.output_folder,
                        model=args.model,
                        n_starts=args.n_starts,
                        show_errors=show_errors,
                    )
                except Exception as e:
                    print(f"ch {ch_idx} XY fit error: {e}")

            if valid_z.any():
                try:
                    courbe_z = np.nanmean(profil_z[valid_z], axis=0)
                    plot_z_autocorrelation(
                        courbe_z,
                        ch_idx,
                        z_scale,
                        use_pixel_units,
                        args.output_folder,
                        model=args.model,
                        n_starts=args.n_starts,
                        show_errors=show_errors,
                    )
                except Exception as e:
                    print(f"ch {ch_idx} Z fit error: {e}")

    print(f"Graphs saved to {args.output_folder}")


if __name__ == "__main__":
    main()
