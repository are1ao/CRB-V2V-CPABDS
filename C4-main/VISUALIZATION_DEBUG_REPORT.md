# Visualization Run Report

## Goal

Verify that the visualization pipeline can run once in the `torch118` conda environment and confirm that the local OpenCOOD code path is healthy enough to load data and render a result.

## Environment

- Conda env: `torch118`
- Python: `3.9.25`
- Entry script: `opencood/visualization/vis_data_sequence.py`
- Dataset used: `v2xset/validate`

## Command Used

```bash
env MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg \
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --color_mode intensity \
  --num_frames 1
```

`MPLCONFIGDIR` and `XDG_CACHE_HOME` were set only to avoid cache warnings in this sandboxed environment.

## Errors Encountered and Fixes

### 1. Open3D window creation failed

Error:

```text
AttributeError: 'NoneType' object has no attribute 'background_color'
```

Cause:
`open3d.visualization.Visualizer().create_window()` could not create a usable window in the current headless environment.

Fix:
- Added a headless fallback in `opencood/visualization/vis_data_sequence.py`.
- When Open3D window creation fails, the script now saves a static preview image instead of crashing.

### 2. Matplotlib fallback tried to use Tk

Error:

```text
_tkinter.TclError: couldn't connect to display ":0"
```

Cause:
The first fallback implementation still relied on a GUI matplotlib backend.

Fix:
- Reworked the fallback rendering in `opencood/visualization/vis_utils.py` to use `FigureCanvasAgg`, which is display-independent.

### 3. Visualization config used a machine-specific path

Problem:
`opencood/hypes_yaml/visualization.yaml` pointed to `/home/wcp/c4/OpenCOOD-main/v2xset/validate`.

Fix:
- Changed `validate_dir` to the portable relative path `v2xset/validate`.

### 4. Script did not exit automatically

Problem:
The original visualization loop was infinite, which is inconvenient for one-shot verification.

Fix:
- Added `--num_frames` to stop after a fixed number of frames.
- Set the visualization dataloader `num_workers` to `0` for a more stable one-shot run in this environment.

## Result

The visualization pipeline ran successfully in fallback mode and generated:

- `opencood/logs/visualization_preview.png`

The output file exists and is a valid PNG image (`1500x600`).

## Files Updated

- `opencood/hypes_yaml/visualization.yaml`
- `opencood/visualization/vis_data_sequence.py`
- `opencood/visualization/vis_utils.py`
