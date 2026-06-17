"""
3D Autocorrelation on cpu with pyfftw

flow of the script:
  - load all .tif slices, downsample, compute tissue mask 
  - find valid tile positions in XY and Z based on tissue coverage
  - fuse consecutive z positions into tall tiles, keep the deepest ones
  - for each tile: compute 3D autocorrelation, then reduce to:
       - radial_2d[dz, r] : autocorrelation as a function of z and radius
       - profil_z[dz] : same but averaged over all radius
  - save one row, per tile, per channel into a autocorrelation_raw.h5 file
"""



import argparse
import atexit
import gc
import os
import pickle
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import psutil
import pyfftw
import pyvips
import skimage.filters
from natsort import natsorted
from numpy.fft import fftshift
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, disk
from tqdm import tqdm

# SLOTH_CACHE_DIR = f"/scratch/users/condesala/.sloth_cache"
SLOTH_CACHE_DIR = "default"

from sloth import sloth_cache

pyvips.cache_set_max(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Autocorrelation Analysis (Cylindrical) — pyfftw CPU"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="/home/arthur/Documents/slices/full_resolution/",
        help="Input folder containing .tif files",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder for results (default: autocorrelation3D relative to input folder)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=1024, help="Tile size in XY dimensions"
    )
    parser.add_argument(
        "--stride-factor", type=int, default=1, help="Stride factor for tile placement"
    )
    parser.add_argument(
        "--xy-scale", type=float, default=0.325, help="XY physical scale (microns)"
    )
    parser.add_argument(
        "--z-scale", type=float, default=24, help="Z physical scale (microns)"
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=10,
        help="Downsample factor for mask and tile plot",
    )
    parser.add_argument(
        "--min-tile-coverage",
        type=float,
        default=0.7,
        help="Minimum tile coverage threshold for a tile to be considered valid",
    )
    parser.add_argument(
        "--max-tile-depth",
        type=int,
        default=None,
        help="Maximum tile depth in Z slices after fusion (default: no limit)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=800,
        help="Maximum number of fused tiles to keep (default: 800)",
    )
    parser.add_argument(
        "--fftw-threads",
        type=int,
        default=None,
        help="Number of threads for pyfftw (default: all available CPUs)",
    )
    parser.add_argument(
        "--fftw-planner",
        type=str,
        default="FFTW_ESTIMATE",
        choices=["FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"],
        help=(
            "pyfftw planner effort level (default: FFTW_MEASURE). I had best results with FFTW_ESTIMATE and FFTW_MEASURE "
            "FFTW_ESTIMATE: fastest to plan, slowest transforms. "
            "FFTW_PATIENT / FFTW_EXHAUSTIVE: slower to plan, fastest transforms "
        ),
    )
    parser.add_argument(
        "--fftw-wisdom-file",
        type=str,
        default=None,
    )
    return parser.parse_args()




# pyfftw first plan on what algorithm to use on our data, and chooses the best one. it stores those analysis in a "wisdom" file
_fftw_plan_cache: dict = {}

def get_fftw_plan(shape: tuple, n_threads: int, planner: str):
    key = (shape, n_threads, planner)
    if key in _fftw_plan_cache:
        return _fftw_plan_cache[key]

    _fftw_plan_cache.clear()
    gc.collect()

    rfft_shape = (
        shape[0],
        shape[1],
        shape[2] // 2 + 1,
    )  # we only need real output so we can divide shape by 2, pyfftw takes care of the rest

    buf_in = pyfftw.empty_aligned(shape, dtype=np.float32)
    buf_out = pyfftw.empty_aligned(rfft_shape, dtype=np.complex64)

    fft_obj = pyfftw.FFTW(
        buf_in,
        buf_out,
        axes=(0, 1, 2),
        direction="FFTW_FORWARD",
        flags=(planner, "FFTW_DESTROY_INPUT"),
        threads=n_threads,
    )
    ifft_obj = pyfftw.FFTW(
        buf_out,
        buf_in,
        axes=(0, 1, 2),
        direction="FFTW_BACKWARD",
        flags=(planner, "FFTW_DESTROY_INPUT"),
        threads=n_threads,
    )

    _fftw_plan_cache[key] = (fft_obj, ifft_obj, buf_in, buf_out)
    return _fftw_plan_cache[key]


def load_wisdom(wisdom_path: str):
    if os.path.exists(wisdom_path):
        with open(wisdom_path, "rb") as fh:
            pyfftw.import_wisdom(pickle.load(fh))
        print(f"Loaded FFTW wisdom from {wisdom_path}")
    else:
        print(f"No wisdom file at {wisdom_path}")


def save_wisdom(wisdom_path: str):
    with open(wisdom_path, "wb") as fh:
        pickle.dump(pyfftw.export_wisdom(), fh)
    print(f"Saved FFTW wisdom to {wisdom_path}")


def vips_to_numpy(img):
    format_to_dtype = {
        "uchar": np.uint8,
        "char": np.int8,
        "ushort": np.uint16,
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "double": np.float64,
    }
    return np.ndarray(
        buffer=img.write_to_memory(),
        dtype=format_to_dtype[img.format],
        shape=[img.height, img.width, img.bands]
        if img.bands > 1
        else [img.height, img.width],
    )


def print_mem():
    process = psutil.Process(os.getpid())
    print(f"RAM: {process.memory_info().rss / 1024**3:.1f} GB")


@sloth_cache(SLOTH_CACHE_DIR, verbose=0) # save the results of the functions on disk to make later runs faster
def compute_roi(mask, downsample_factor=10):
    mask_np = mask.compute() if hasattr(mask, "compute") else mask

    if not np.any(mask_np):
        return None

    coords = np.argwhere(mask_np)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    xmin = np.int64(x_min * downsample_factor * 0.95)
    ymin = np.int64(y_min * downsample_factor * 0.95)
    xmax = np.int64((x_max + 1) * downsample_factor * 1.05)
    ymax = np.int64((y_max + 1) * downsample_factor * 1.05)

    return (xmin, ymin, xmax, ymax)


@sloth_cache(SLOTH_CACHE_DIR, verbose=0)
def preprocess_slice(file_path, downsample_factor):
    img = pyvips.Image.new_from_file(file_path, access="sequential")
    shrunk = img.shrink(downsample_factor, downsample_factor)
    sample = vips_to_numpy(shrunk)

    if sample.ndim == 2:
        sample = sample[..., np.newaxis]

    mean_sample = sample.mean(axis=2)
    threshold = skimage.filters.threshold_otsu(mean_sample)
    mask = binary_fill_holes(mean_sample > threshold)
    se_radius = max(1, int(round(200 / downsample_factor)))
    mask = closing(mask, disk(se_radius)).astype(bool)

    roi = compute_roi(mask, downsample_factor=downsample_factor)
    return mask, roi


# precompute for each pixel: which radial ring it belongs to. reused across all tiles to avoid recomputing
def get_precomputed_radial_grid(height: int, width: int):
    y_center, x_center = height // 2, width // 2
    y_idx, x_idx = np.indices((height, width))

    r_xy = np.round(np.sqrt((x_idx - x_center) ** 2 + (y_idx - y_center) ** 2)).astype(
        np.intp
    )
    r_flat = r_xy.ravel()

    max_r = width // 2
    counts = np.bincount(r_flat)[: max_r + 1]  # how many pixels fall in each ring
    valid = counts > 0

    return r_flat, counts, valid, max_r


def process_tile_cylindrical(
    volume: np.ndarray,
    fftw_threads,
    fftw_planner: str,
    r_flat: np.ndarray,
    counts: np.ndarray,
    valid: np.ndarray,
    max_r: int,
) -> tuple[np.ndarray, np.ndarray]:

    depth, height, width = volume.shape

    fft_obj, ifft_obj, buf_in, buf_out = get_fftw_plan(
        (depth, height, width), fftw_threads, fftw_planner
    )

    buf_in[:] = volume
    buf_in -= buf_in.mean(dtype=np.float64) 

    # forward FFT, multiply by conjugate, inverse FFT
    fft_obj.execute()
    buf_out *= np.conj(buf_out)
    ifft_obj.execute()

    # normalize so that autocorrelation at zero shift = 1
    max_val = buf_in[0, 0, 0]
    if max_val > 0:
        buf_in /= max_val

    z_center = depth // 2
    max_dz = min(z_center, depth - 1 - z_center)

    # z profil

    z_sums = buf_in.sum(axis=(1, 2))
    profil_z = np.zeros(max_dz + 1, dtype=np.float32)
    profil_z[0] = z_sums[0]
    for dz in range(1, max_dz + 1):
        profil_z[dz] = (z_sums[dz] + z_sums[-dz]) / 2.0

    # radial map
    radial_2d = np.full((max_dz + 1, max_r + 1), np.nan, dtype=np.float32)
    for dz in range(max_dz + 1):
        if dz == 0:
            slice_AC = np.fft.fftshift(buf_in[0])
        else:
            slice_AC = np.fft.fftshift((buf_in[dz] + buf_in[-dz]) / 2.0)

        sums = np.bincount(r_flat, weights=slice_AC.ravel())[: max_r + 1]
        radial_2d[dz, valid] = sums[valid] / counts[valid]

    return radial_2d, profil_z


# fuse the tiles for the highest tile size possible (or to the defined max depth)
def fuse_tiles(
    valid_positions, tile_depth: int, z_stride: int, max_tile_depth=None
) -> list:
    grouped: dict = defaultdict(list)
    for x, y, z in valid_positions:
        grouped[(x, y)].append(z)

    fused = []
    for (x, y), z_list in grouped.items():
        z_set = sorted(set(z_list))
        if not z_set:
            print(f"no z positions for tile at ({x}, {y}) ?")
            continue

        start = prev = z_set[0]
        for z in z_set[1:]:
            if z == prev + z_stride:
                prev = z
            else:
                fused.append((x, y, start, prev + tile_depth))
                start = prev = z
        fused.append((x, y, start, prev + tile_depth))

    if max_tile_depth is None or max_tile_depth <= 0:
        return fused

    limited = []
    for x, y, z0, z1 in fused:
        current = z0
        while current < z1:
            z_end = min(current + max_tile_depth, z1)
            limited.append((x, y, current, z_end))
            current = z_end

    return limited

# extract a 3D tile from the volume
def extract_cube(
    x: int,
    y: int,
    z0: int,
    z1: int,
    input_folder: str,
    files_list: list,
    tile_size: int,
) -> np.ndarray:
    cube_slices = []
    for z in range(z0, z1):
        full_path = os.path.join(input_folder, files_list[z])
        img = pyvips.Image.new_from_file(full_path, access="sequential")
        tile_vips = img.crop(x, y, tile_size, tile_size)
        tile_np = vips_to_numpy(tile_vips)
        del img, tile_vips
        if tile_np.ndim == 3 and tile_np.shape[-1] == 1:
            tile_np = tile_np.squeeze(-1)
        cube_slices.append(tile_np)
    return np.stack(cube_slices, axis=0)


def main():
    args = parse_args()

    INPUT_FOLDER = args.input_folder
    TILE_SIZE = args.tile_size
    STRIDE_FACTOR = args.stride_factor
    XY_PHYSICAL_SCALE = args.xy_scale
    Z_PHYSICAL_SCALE = args.z_scale
    DOWNSAMPLE_FACTOR = args.downsample_factor
    MIN_TILE_COVERAGE = args.min_tile_coverage
    MAX_TILE_DEPTH = args.max_tile_depth
    MAX_TILES = args.max_tiles
    FFTW_THREADS = args.fftw_threads or os.cpu_count()
    FFTW_PLANNER = args.fftw_planner

    if args.output_folder:
        OUTPUT_FOLDER = args.output_folder
    else:
        OUTPUT_FOLDER = os.path.join(
            os.path.dirname(INPUT_FOLDER.rstrip("/")), "autocorrelation3D"
        )
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    WISDOM_PATH = args.fftw_wisdom_file or os.path.join(
        OUTPUT_FOLDER, "fftw_wisdom.pkl"
    )
    load_wisdom(WISDOM_PATH)
    atexit.register(save_wisdom, WISDOM_PATH)

    print(f"pyfftw: {FFTW_THREADS} threads, planner={FFTW_PLANNER}")

    # masks and rois
    print("Reading file paths...")
    files = natsorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tif")])
    file_paths = [os.path.join(INPUT_FOLDER, f) for f in files]
    print(f"{len(file_paths)} files to process")

    with ProcessPoolExecutor(max_workers=FFTW_THREADS) as executor:
        results = list(
            tqdm(
                executor.map(
                    preprocess_slice, file_paths, [DOWNSAMPLE_FACTOR] * len(file_paths)
                ),
                total=len(file_paths),
                desc="Preprocessing slices",
            )
        )

    masks, rois = zip(*results)
    masks, rois = list(masks), list(rois)
    del results
    print(f"Computed {len(rois)} ROIs")

    valid_rois = [r for r in rois if r is not None]
    mins = np.min(valid_rois, axis=0)
    maxs = np.max(valid_rois, axis=0)
    xmin_global = int(mins[0])
    ymin_global = int(mins[1])
    xmax_global = int(maxs[2])
    ymax_global = int(maxs[3])
    print(f"Global ROI: ({xmin_global}, {ymin_global}, {xmax_global}, {ymax_global})")

    # get valid tile positions
    stride = TILE_SIZE // STRIDE_FACTOR
    tile_depth_slices = max(
        1, int(round((TILE_SIZE * XY_PHYSICAL_SCALE) / Z_PHYSICAL_SCALE) // 8)
    )
    z_stride = max(1, tile_depth_slices // 2)

    mask_volume_ds = np.stack(masks, axis=0)
    del masks
    gc.collect()
    t_ds_xy = TILE_SIZE // DOWNSAMPLE_FACTOR
    y_max_ds = mask_volume_ds.shape[1]
    x_max_ds = mask_volume_ds.shape[2]
    start_y = (ymin_global // stride) * stride
    start_x = (xmin_global // stride) * stride

    print("Enumerating valid tile positions...")
    valid_positions: set = set()
    for z in range(0, len(file_paths) - tile_depth_slices + 1, z_stride):
        z_end = z + tile_depth_slices
        for y in range(start_y, ymax_global - TILE_SIZE + 1, stride):
            for x in range(start_x, xmax_global - TILE_SIZE + 1, stride):
                y_ds = y // DOWNSAMPLE_FACTOR
                x_ds = x // DOWNSAMPLE_FACTOR
                y_ds_end = y_ds + t_ds_xy
                x_ds_end = x_ds + t_ds_xy

                if y_ds_end > y_max_ds or x_ds_end > x_max_ds:
                    continue

                mask_tile_3d = mask_volume_ds[z:z_end, y_ds:y_ds_end, x_ds:x_ds_end]
                covered = np.count_nonzero(mask_tile_3d)
                if covered >= (MIN_TILE_COVERAGE * mask_tile_3d.size):
                    valid_positions.add((x, y, z))

    print(f"3D valid tile count: {len(valid_positions)}")
    print(
        f"Tile shape (x, y, z slices): ({TILE_SIZE}, {TILE_SIZE}, {tile_depth_slices})"
    )

    # merge consecutive z positions into tall tiles to maximize depth per tile
    fused_tiles = fuse_tiles(
        valid_positions,
        tile_depth_slices,
        z_stride,
        max_tile_depth=MAX_TILE_DEPTH,
    )

    MIN_TILE_DEPTH = 10

    # drop tiles too shallow for a meaningful autocorrelation
    fused_tiles = [
        (x, y, z0, z1) for x, y, z0, z1 in fused_tiles if (z1 - z0) >= MIN_TILE_DEPTH
    ]

    # keep the deepest tiles first 
    total_fused = len(fused_tiles)
    if MAX_TILES is None or MAX_TILES <= 0:
        fused_tiles = sorted(fused_tiles, key=lambda t: t[3] - t[2], reverse=True)
    else:
        fused_tiles = sorted(fused_tiles, key=lambda t: t[3] - t[2], reverse=True)[
            :MAX_TILES
        ]

    discarded = total_fused - len(fused_tiles)
    print(f"Fused tiles kept: {len(fused_tiles)}/{total_fused} (discarded {discarded})")

    # Free large mask arrays from ram before processing.
    del mask_volume_ds, rois, valid_rois, valid_positions
    gc.collect()

    # creating output file
    filepath = os.path.join(OUTPUT_FOLDER, "autocorrelation_raw.h5")
    total_tiles = len(fused_tiles)
    max_z_len = max(z1 - z0 for _, _, z0, z1 in fused_tiles)
    max_d_dim = (max_z_len // 2) + 1
    max_r_dim = (TILE_SIZE // 2) + 1

    print("Creating output file...")
    with h5py.File(filepath, "w", libver="latest") as f:
        f.attrs["tile_size"] = TILE_SIZE
        f.attrs["stride_factor"] = STRIDE_FACTOR
        f.attrs["xy_scale"] = XY_PHYSICAL_SCALE
        f.attrs["z_scale"] = Z_PHYSICAL_SCALE
        f.attrs["downsample_factor"] = DOWNSAMPLE_FACTOR
        f.attrs["min_tile_coverage"] = MIN_TILE_COVERAGE
        f.attrs["input_folder"] = INPUT_FOLDER
        f.attrs["z_stride"] = z_stride
        f.attrs["tile_depth_slices"] = tile_depth_slices
        f.attrs["fftw_threads"] = FFTW_THREADS
        f.attrs["fftw_planner"] = FFTW_PLANNER
        if MAX_TILE_DEPTH is not None:
            f.attrs["max_tile_depth"] = MAX_TILE_DEPTH
        if MAX_TILES is not None and MAX_TILES > 0:
            f.attrs["max_tiles"] = MAX_TILES
        f.create_dataset("fused_tiles", data=np.array(fused_tiles, dtype=np.int32))

        for ch_idx in range(3):
            g = f.create_group(f"channel_{ch_idx}")
            g.create_dataset(
                "radial_2d",
                shape=(total_tiles, max_d_dim, max_r_dim),
                dtype=np.float32,
                fillvalue=np.nan,
                chunks=(1, max_d_dim, max_r_dim),
            )
            g.create_dataset(
                "profil_z",
                shape=(total_tiles, max_d_dim),
                dtype=np.float32,
                fillvalue=np.nan,
                chunks=(1, max_d_dim),
            )

    tile_count = {0: 0, 1: 0, 2: 0}

    print("Precomputing radial grid...")
    r_flat, counts, valid, max_r = get_precomputed_radial_grid(TILE_SIZE, TILE_SIZE)
    
    # group tiles by depth so the fftw plan is reused within each group
    groups = defaultdict(list)
    for x, y, z0, z1 in fused_tiles:
        depth = z1 - z0
        groups[depth].append((x, y, z0, z1))

    # shuffle tiles inside each groups, so that if we only take a small sample of all tiles, they are spread across the whole volume
    for depth in groups:
        random.shuffle(groups[depth])

    sorted_groups = sorted(groups.items(), key=lambda item: item[0], reverse=True)

    all_tiles = [
        (depth, x, y, z0, z1)
        for depth, tiles in sorted_groups
        for x, y, z0, z1 in tiles
    ]

    if MAX_TILES is not None and MAX_TILES > 0:
        all_tiles = all_tiles[:MAX_TILES]

    print(f"Starting tile processing ({len(sorted_groups)} unique depth groups)...")

    with h5py.File(filepath, "a", libver="latest") as f:
        f.swmr_mode = True
        pbar = tqdm(all_tiles, total=total_tiles, desc="Processing")

        for depth, x, y, z0, z1 in pbar:
            pbar.set_postfix(depth=depth)
            cube = extract_cube(x, y, z0, z1, INPUT_FOLDER, files, TILE_SIZE)

            # process each channel independently
            for ch_idx, volume_canal in enumerate(cube.transpose(3, 0, 1, 2)):
                heatmap_2d, profil_z = process_tile_cylindrical(
                    volume_canal,
                    FFTW_THREADS,
                    FFTW_PLANNER,
                    r_flat,
                    counts,
                    valid,
                    max_r,
                )

                ch_group = f[f"channel_{ch_idx}"]
                count = tile_count[ch_idx]
                d, r = heatmap_2d.shape

                ch_group["radial_2d"][count, :d, :r] = heatmap_2d
                ch_group["profil_z"][count, :d] = profil_z
                tile_count[ch_idx] += 1

                f.flush()

            del cube

        print(f"processed tiles : {tile_count}")

    print("Done.")


if __name__ == "__main__":
    main()
