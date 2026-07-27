# bootcamp_utils — the ONE shared library for all PYNQ Bootcamp notebooks

This folder is a single, shared Python library used by **every** session notebook
in `bootcamp_sessions/`. It holds the "boilerplate" and "complex" code — the parts
that were the same (copy-pasted) across notebooks and that a 6th–12th grader
shouldn't have to write from scratch. Now each notebook just imports these
friendly, well-named helpers.

## How a notebook uses it

The library is **installed as a real Python package**, so notebooks import it
directly — no `sys.path` lines, no per-notebook setup:

```python
from bootcamp_utils import load_dpu, setup_dpu_buffers, run_model, classify
```

## Zero-touch setup on a freshly flashed KV260 image

This works immediately after flashing a fresh image because the install is
automated (see the files in the parent folder, `bootcamp_sessions/`):

- `pyproject.toml` — makes `bootcamp_utils` a proper installable package.
- `install.sh` — does an **editable** install (`pip install -e .`) so edits to
  the library are picked up live, and also drops an IPython startup fallback
  that adds this folder to the path at kernel start (works even offline).
- `bootcamp-utils.service` — a systemd one-shot unit that runs `install.sh` on
  first boot, so no human has to run anything.

Enable it once when building the image:

```bash
# from bootcamp_sessions/
sudo cp bootcamp-utils.service /etc/systemd/system/
sudo systemctl enable bootcamp-utils.service
# (or just run: bash install.sh)
```

Because the install is editable and the service is idempotent, this scales to
any number of boards: flash → boot → notebooks just work.

## What's inside

| Module | What it gives you | Used by |
|--------|-------------------|---------|
| `dpu.py` | `load_dpu`, `setup_dpu_buffers`, `setup_dpu_buffers_multi`, `run_model`, `softmax`, `classify` | 101, 201, 301, 503 |
| `hardware.py` | `setup_led_bar`, `set_led_bar`, `setup_oled`, `show_on_oled` | 101, 201, 301 |
| `vision.py` | `read_image`, `show_image`, `plot_images`, `open_camera`, `capture_frame`, `list_images` | 201, 501, 503 |
| `color_utils.py` | `load_classes`, `make_colors` | 301 |
| `live_detection.py` | `launch_live_detection` (threaded real-time webcam loop) | 301 |
| `pose.py` | `preprocess_pose_image`, `make_heatmap`, `compare_poses` | 503 |

Everything is exported from the top level, so `from bootcamp_utils import X` works
for any helper above.

## Common functions shared between notebooks

These are the functions that appeared (in slightly different forms) in multiple
notebooks and are now written **once** here:

- **Loading the FPGA + model** — every AI notebook did `DpuOverlay("dpu.bit")`
  then `overlay.load_model(...)`. Now: `overlay = load_dpu("model.xmodel")`.
- **DPU buffer setup** — every AI notebook allocated input/output NumPy buffers
  from the model's tensor shapes. Now: `setup_dpu_buffers(overlay)`.
- **Running the model** — `execute_async` + `wait` + reshape. Now: `run_model(...)`.
- **Softmax / argmax** — the MNIST classification helper. Now: `classify(...)`.
- **Grove LED Bar** — 101 and 201 set it up identically. Now: `setup_led_bar` / `set_led_bar`.
- **Webcam capture** — 201, 503 opened the camera the same way. Now: `open_camera` / `capture_frame`.
- **Side-by-side image plots** — 501's `plot_images` is now shared.

## For students

You can (and should!) open these files and read them to see how the "magic"
works. You won't need to edit them to finish a session — just call the helpers.

## For instructors

- Keep this as the **single source of truth**. Don't copy it into session folders.
- The version is tracked in `__init__.py` (`__version__`).
