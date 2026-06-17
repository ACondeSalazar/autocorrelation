# Autocorrelation 3D

## Setup

```bash
python -m venv .venv # Python 3.11 recommended
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `pyvips` requires `libvips` installed system-wide.
>
> - **Windows:** `pip install pyvips-binary` (bundles libvips, no separate install needed)
> - **Ubuntu/Debian:** `sudo apt install libvips-dev`

## Usage

```bash
python autocorrelation3D_cylindrical_cpu.py \
  --input-folder /path/to/tif/slices/ \
  --output-folder /path/to/output/ \        # default: autocorrelation3D/ next to input folder
  --tile-size 1024 \                        # XY tile size in pixels
  --stride-factor 1 \                       # tile stride = tile-size / stride-factor, overlapping tiles overlap
  --xy-scale 0.325 \                        # microns per pixel (XY)
  --z-scale 24 \                            # microns per slice (Z)
  --downsample-factor 10 \                  # for tissue mask computation only
  --min-tile-coverage 0.7 \                 # min fraction of tile covered by tissue
  --max-tile-depth None \                   # max Z slices per fused tile (no limit)
  --max-tiles 800 \                         # max number of tiles to process
  --fftw-threads None \                     # default: all CPUs
  --fftw-planner FFTW_ESTIMATE \            # FFTW_ESTIMATE | FFTW_MEASURE | FFTW_PATIENT | FFTW_EXHAUSTIVE
  --fftw-wisdom-file /path/to/wisdom.pkl    # optional: reuse FFTW planning across runs
```
