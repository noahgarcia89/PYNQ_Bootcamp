# PYNQ 301 - Object Detection

Real-time object detection using YOLOv3 on the KRIA KV260 board.

## Files Structure

```
PYNQ 301 - Object Detection/
├── PYNQ 301 - Object Detection.ipynb  # Main notebook
├── PYNQ-301_Object_Detection.pptx     # Presentation slides
├── yolo_helpers.py                     # YOLO-specific helpers
├── class_explorer.py                   # Interactive class filtering widget
├── confidence_explorer.py              # Confidence threshold widget
├── img/                                # Sample images & class names
│   └── voc_classes.txt                # VOC dataset class names
└── tf_yolov3_voc.xmodel               # YOLOv3 model weights
```

Generic helpers (colors, classes, DPU buffers, image listing, live detection)
now live in the **ONE shared library** at `../bootcamp_utils/`, which every
bootcamp session shares and which is installed as a package (see
`../install.sh`). See `../bootcamp_utils/README.md`.

## Quick Start

1. Open `PYNQ 301 - Object Detection.ipynb`
2. Run cells in order — `bootcamp_utils` is already installed, so imports just work
   (no `sys.path` lines needed).

## Import Flow

```
Notebook import cell:
  ├─> from bootcamp_utils import load_classes, make_colors
  ├─> from bootcamp_utils import setup_dpu_buffers, list_images
  ├─> from bootcamp_utils import launch_live_detection
  └─> from yolo_helpers import load_anchors, pre_process, evaluate, draw_boxes
```

`yolo_helpers.py` re-exports shared `bootcamp_utils` functions for convenience
(and uses `setup_dpu_buffers_multi` under the hood since YOLO has 3 outputs).
