"""
Old script to compute autocorrelation on gpu 
"""

import argparse
import gc
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import h5py
import psutil
import pyvips
import skimage.filters
from natsort import natsorted
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, disk
from tqdm import tqdm

# SLOTH_CACHE_DIR = f"/scratch/users/condesala/.sloth_cache"
SLOTH_CACHE_DIR = f"default"

from sloth import sloth_cache

pyvips.cache_set_max(0)

USE_GPU = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Autocorrelation Analysis (Cylindrical)"
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
        help="Minimum tile coverage threshold",
    )
    parser.add_argument(
        "--max-tile-depth",
        type=int,
        default=None,
        help="Maximum tile depth in Z slices after fusion (default: no limit)",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode",
    )

    args = parser.parse_args()

    global USE_GPU
    USE_GPU = not args.cpu

    return args


import numpy as np

args = parse_args()

if USE_GPU:
    import cupy as cp
    from cupyx.scipy.fft import fftshift, irfftn, rfftn

    print("Using cupy and GPU")
else:
    import numpy as cp
    from scipy.fft import fftshift, irfftn, rfftn

    print("Using numpy and CPU")


INPUT_FOLDER = args.input_folder
TILE_SIZE = args.tile_size
STRIDE_FACTOR = args.stride_factor
XY_PHYSICAL_SCALE = args.xy_scale
Z_PHYSICAL_SCALE = args.z_scale
DOWNSAMPLE_FACTOR = args.downsample_factor
MIN_TILE_COVERAGE = args.min_tile_coverage
MAX_TILE_DEPTH = args.max_tile_depth

# Create output folder
if args.output_folder:
    OUTPUT_FOLDER = args.output_folder
else:
    OUTPUT_FOLDER = os.path.join(
        os.path.dirname(INPUT_FOLDER.rstrip("/")), "autocorrelation3D"
    )
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

Z_RATIO = Z_PHYSICAL_SCALE / XY_PHYSICAL_SCALE


max_r = TILE_SIZE // 2
y_idx, x_idx = cp.indices((TILE_SIZE, TILE_SIZE))
center = (TILE_SIZE / 2) - 0.5
r_template = cp.round(cp.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2)).astype(
    int
)


def free_mem():
    if USE_GPU:
        cp.cuda.Stream.null.synchronize()
        cp._default_memory_pool.free_all_blocks()

    gc.collect()


def to_numpy(arr):
    if USE_GPU:
        return cp.asnumpy(arr)
    else:
        return np.asarray(arr)


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


@sloth_cache(SLOTH_CACHE_DIR, verbose=0)
def compute_roi(mask, downsample_factor=10):
    if hasattr(mask, "compute"):
        mask_np = mask.compute()
    else:
        mask_np = mask

    if not np.any(mask_np):
        return None

    coords = np.argwhere(mask_np)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    ymin = y_min * downsample_factor
    xmin = x_min * downsample_factor
    ymax = (y_max + 1) * downsample_factor
    xmax = (x_max + 1) * downsample_factor

    ymin = np.int64(ymin * 0.95)
    xmin = np.int64(xmin * 0.95)
    ymax = np.int64(ymax * 1.05)
    xmax = np.int64(xmax * 1.05)

    return (xmin, ymin, xmax, ymax)


@sloth_cache(SLOTH_CACHE_DIR, verbose=0)
def preprocess_slice(file_path):
    img = pyvips.Image.new_from_file(file_path, access="sequential")

    shrunk = img.shrink(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)
    sample = vips_to_numpy(shrunk)

    if sample.ndim == 2:
        sample = sample[..., np.newaxis]

    mean_sample = sample.mean(axis=2)
    threshold = skimage.filters.threshold_otsu(mean_sample)
    mask = binary_fill_holes(mean_sample > threshold)
    se_radius = max(1, int(round(200 / DOWNSAMPLE_FACTOR)))
    mask = closing(mask, disk(se_radius)).astype(bool)

    roi = compute_roi(mask, downsample_factor=DOWNSAMPLE_FACTOR)
    return mask, roi


def process_tile_cylindrical(volume):
    vol = cp.asarray(volume, dtype=cp.float32)
    vol -= cp.mean(vol)

    depth, height, width = vol.shape
    padded_depth = depth

    # fft
    F = rfftn(vol, s=(padded_depth, height, width))
    del vol
    free_mem()

    power_spectrum = cp.abs(F) ** 2
    del F
    free_mem()

    AC = irfftn(power_spectrum, s=(padded_depth, height, width))
    del power_spectrum
    free_mem()
    # print(AC.shape)

    AC = fftshift(AC)

    max_val = cp.max(AC)
    if max_val > 0:
        AC /= max_val

    z_center, y_center, x_center = padded_depth // 2, height // 2, width // 2

    max_dz = min(z_center, depth - 1 - z_center)

    # radius and theta integral
    profil_z = cp.sum(AC, axis=(1, 2))[z_center : z_center + max_dz + 1]

    # distances
    y_idx, x_idx = cp.indices((height, width))
    r_xy = cp.round(cp.sqrt((x_idx - x_center) ** 2 + (y_idx - y_center) ** 2)).astype(
        int
    )

    max_r = width // 2
    r_flat = r_xy.ravel()

    counts = cp.bincount(r_flat)[: max_r + 1]
    valid = counts > 0

    radial_2d = cp.full((max_dz + 1, max_r + 1), cp.nan)

    for dz in range(max_dz + 1):
        if dz == 0:
            slice_AC = AC[z_center]
        else:
            slice_AC = (AC[z_center + dz] + AC[z_center - dz]) / 2.0

        sums = cp.bincount(r_flat, weights=slice_AC.ravel())[: max_r + 1]
        radial_2d[dz, valid] = sums[valid] / counts[valid]

    radial_2d_np = to_numpy(radial_2d)
    profil_z_np = to_numpy(profil_z)

    free_mem()

    return radial_2d_np, profil_z_np


# fuse tiles along z axis
def fuse_tiles(valid_positions, tile_depth, z_stride, max_tile_depth=None):
    grouped = defaultdict(list)

    for x, y, z in valid_positions:
        grouped[(x, y)].append(z)

    fused = []

    for (x, y), z_list in grouped.items():
        z_set = sorted(set(z_list))

        if not z_set:
            print(f"no tiles in {(x, y)}")

        start = z_set[0]
        prev = z_set[0]

        for z in z_set[1:]:
            if z == prev + z_stride:
                prev = z

            else:
                fused.append((x, y, start, prev + tile_depth))
                start = z
                prev = z

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


def print_mem():
    process = psutil.Process(os.getpid())
    print(f"RAM: {process.memory_info().rss / 1024**3:.1f} GB")


def extract_cube(x, y, z0, z1, files_list):
    cube_slices = []
    for z in range(z0, z1):
        full_path = os.path.join(INPUT_FOLDER, files_list[z])
        img = pyvips.Image.new_from_file(full_path, access="sequential")
        tile_vips = img.crop(x, y, TILE_SIZE, TILE_SIZE)
        tile_np = vips_to_numpy(tile_vips)
        del img, tile_vips
        if tile_np.ndim == 3 and tile_np.shape[-1] == 1:
            tile_np = tile_np.squeeze(-1)
        cube_slices.append(tile_np)
    return np.stack(cube_slices, axis=0)


def triple_exp(r, A, B, C, l1, l2, l3):
    return A * np.exp(-r / l1) + B * np.exp(-r / l2) + C * np.exp(-r / l3)


def main():

    # make mask and rois

    print("Reading file paths...")
    files = natsorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tif")])
    file_paths = [os.path.join(INPUT_FOLDER, f) for f in files]

    print(f"{len(file_paths)} files to process")

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(preprocess_slice, file_paths))

    masks, rois = zip(*results)
    masks, rois = list(masks), list(rois)
    del results

    print(f"Computed {len(rois)} ROIs")

    mins = np.min([r for r in rois if r is not None], axis=0)
    maxs = np.max([r for r in rois if r is not None], axis=0)

    xmin_global, ymin_global = mins[0], mins[1]
    xmax_global, ymax_global = maxs[2], maxs[3]

    print(f"Global ROI : {(xmin_global, ymin_global, xmax_global, ymax_global)}")

    # Make tiles

    stride = TILE_SIZE // STRIDE_FACTOR
    valid_positions = set()

    start_y = (ymin_global // stride) * stride
    start_x = (xmin_global // stride) * stride

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

    print("Extracting tiles")

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
        f"3D tile shape (x, y, z slices): ({TILE_SIZE}, {TILE_SIZE}, {tile_depth_slices})"
    )

    fused_tiles = fuse_tiles(
        valid_positions,
        tile_depth_slices,
        z_stride,
        max_tile_depth=MAX_TILE_DEPTH,
    )
    print(f"Fused tile count: {len(fused_tiles)}")

    # Free large mask arrays before FFT-heavy processing.
    del mask_volume_ds, masks, rois, results, valid_positions
    gc.collect()

    print("Sorting by z size for batching...")
    groups = defaultdict(list)
    for x, y, z0, z1 in fused_tiles:
        depth = z1 - z0
        groups[depth].append((x, y, z0, z1))

    print("Creating output file")

    filepath = os.path.join(OUTPUT_FOLDER, "autocorrelation_raw.h5")

    # Calcul des dimensions maximales globales pour la pré-allocation
    total_tiles = len(fused_tiles)
    max_z_len = max([z1 - z0 for _, _, z0, z1 in fused_tiles])
    global_z_center = max_z_len // 2
    global_max_dz = min(global_z_center, max_z_len - 1 - global_z_center)
    max_d_dim = global_max_dz + 1
    max_r_dim = (TILE_SIZE // 2) + 1

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
        if MAX_TILE_DEPTH is not None:
            f.attrs["max_tile_depth"] = MAX_TILE_DEPTH
        f.create_dataset("fused_tiles", data=np.array(fused_tiles, dtype=np.int32))

        # Pré-allocation de la taille finale exacte
        for ch_idx in [0, 1, 2]:
            ch_group = f.create_group(f"channel_{ch_idx}")
            ch_group.create_dataset(
                "radial_2d",
                shape=(total_tiles, max_d_dim, max_r_dim),
                dtype=np.float32,
                fillvalue=np.nan,
                chunks=(1, max_d_dim, max_r_dim),
            )
            ch_group.create_dataset(
                "profil_z",
                shape=(total_tiles, max_d_dim),
                dtype=np.float32,
                fillvalue=np.nan,
                chunks=(1, max_d_dim),
            )

    tile_count = {0: 0, 1: 0, 2: 0}
    print("Starting batches...")
    print(f"Total tiles to process: {len(groups)} depth groups")

    MIN_DEPTH = 10
    all_tiles = [
        (depth, x, y, z0, z1)
        for depth, tiles in groups.items()
        for x, y, z0, z1 in tiles
        if depth >= MIN_DEPTH
    ]
    # Sort tiles by depth in descending order
    all_tiles = sorted(
        [
            (depth, x, y, z0, z1)
            for depth, tiles in groups.items()
            for x, y, z0, z1 in tiles
            if depth >= MIN_DEPTH
        ],
        key=lambda x: x[0],
        reverse=True,
    )
    all_tiles = all_tiles[:1000]
    total_tiles = len(all_tiles)

    with h5py.File(filepath, "a", libver="latest") as f:
        f.swmr_mode = True
        for i, (depth, x, y, z0, z1) in enumerate(
            tqdm(all_tiles, total=total_tiles), start=1
        ):
            # print(f"Processing tile {i}/{total_tiles}: depth={depth}, x={x}, y={y}, z=({z0},{z1})")

            # print("extracted tile")
            cube = extract_cube(x, y, z0, z1, file_paths)
            for ch_idx, volume_canal in enumerate(cube.transpose(3, 0, 1, 2)):
                # print("processing tile")
                heatmap_2d, profil_z = process_tile_cylindrical(volume_canal)
                free_mem()

                ch_group = f[f"channel_{ch_idx}"]
                count = tile_count[ch_idx]
                d, r = heatmap_2d.shape

                ch_group["radial_2d"][count, :d, :r] = heatmap_2d
                ch_group["profil_z"][count, :d] = profil_z

                tile_count[ch_idx] += 1

                f.flush()
            # print(f"Finished tile {i}/{total_tiles}")

    print("Done")


if __name__ == "__main__":
    main()
