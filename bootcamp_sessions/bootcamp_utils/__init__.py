"""
bootcamp_utils — ONE shared helper library for the whole PYNQ AI Bootcamp.

This single package lives in `bootcamp_sessions/bootcamp_utils/` and is shared by
EVERY session notebook (PYNQ 101, 201, 301, 501, 503, ...). It collects the
"boilerplate" and "complex" code — the parts that are the same in every notebook
and that a 6th–12th grader shouldn't have to write from scratch — into a few
friendly, well-named functions.

How a notebook uses it
----------------------
Each notebook adds the shared folder to the import path with one line, then
imports the helpers it needs:

    import sys; sys.path.insert(0, "..")          # find the shared library
    from bootcamp_utils import load_dpu, run_model, setup_led_bar

The modules
-----------
- dpu.py         : load the FPGA overlay + model, set up DPU buffers, run the
                   model, and turn model outputs into an answer (softmax/argmax).
- hardware.py    : Grove LED bar and Grove OLED setup + easy control.
- vision.py      : read/show images, side-by-side plots, and webcam capture.
- color_utils.py : load class-name lists and build a color for each class.
- live_detection.py : the full real-time webcam detection loop (threaded).

Everything here is meant to be READ by students to see how it works, but they
should not need to edit it to complete a session.
"""
__version__ = "1.0.0"

# ── Convenience re-exports so notebooks can do `from bootcamp_utils import X` ──
from .dpu import (
    load_dpu,
    setup_dpu_buffers,
    setup_dpu_buffers_multi,
    run_model,
    softmax,
    classify,
)
from .hardware import (
    setup_led_bar,
    set_led_bar,
    setup_oled,
    show_on_oled,
)
from .vision import (
    read_image,
    show_image,
    plot_images,
    open_camera,
    capture_frame,
    list_images,
)
from .color_utils import load_classes, make_colors
from .live_detection import launch_live_detection
from .pose import preprocess_pose_image, make_heatmap, compare_poses

__all__ = [
    # dpu
    "load_dpu", "setup_dpu_buffers", "setup_dpu_buffers_multi",
    "run_model", "softmax", "classify",
    # hardware
    "setup_led_bar", "set_led_bar", "setup_oled", "show_on_oled",
    # vision
    "read_image", "show_image", "plot_images", "open_camera", "capture_frame",
    "list_images",
    # color_utils
    "load_classes", "make_colors",
    # live_detection
    "launch_live_detection",
    # pose
    "preprocess_pose_image", "make_heatmap", "compare_poses",
]
