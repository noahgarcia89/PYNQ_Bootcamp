"""
pose.py — shared helpers for the Pose Detection session (PYNQ 503).

Pose detection runs the OpenPose AI model on the FPGA and produces a "heatmap"
that lights up where a person's body is. Getting an image ready for OpenPose
(padding + center-cropping to the exact size the model wants) and comparing two
heatmaps to see if two poses match is fiddly math — so it lives here.

These functions need to know the model's input shape, so you pass in `shape_in`
(from setup_dpu_buffers) when you call them.
"""
import cv2
import numpy as np


def preprocess_pose_image(img, shape_in, pad=40):
    """
    Get an image ready for the OpenPose model.

    Steps: convert to BGR (what OpenPose wants), add a border of padding, then
    center-crop it to exactly the size the model expects, and scale the pixel
    values to be between 0 and 1.

    Parameters
    ----------
    img : the input image (a NumPy array).
    shape_in : the model input shape from setup_dpu_buffers().
    pad : how many pixels of border to add before cropping (default 40).

    Returns
    -------
    img_cropped : the ready-to-use image, shaped for the model.

    Example
    -------
        ready = preprocess_pose_image(frame, shape_in)
    """
    # OpenPose expects BGR color order
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = img.shape[0], img.shape[1]

    # Add a padding border so the person isn't cut off at the edges
    canvas = 255 * np.ones((height + 2 * pad, width + 2 * pad, 3), dtype=img.dtype)
    canvas[pad:height + pad, pad:width + pad, :] = img
    img = canvas

    # Center-crop to exactly the model's input width and height
    left = int(img.shape[1] / 2 - shape_in[2] / 2)
    right = left + shape_in[2]
    top = int(img.shape[0] / 2 - shape_in[1] / 2)
    bottom = top + shape_in[1]

    img_cropped = img[top:bottom, left:right, :]
    img_cropped = img_cropped.astype(np.float32) / 255.0   # scale to 0..1
    return img_cropped


def make_heatmap(model_output, shape_in):
    """
    Turn the raw model output into a heatmap image you can display.

    The model gives many layers; we take the strongest value at each spot and
    resize it to match the input image so it lines up when overlaid.

    Parameters
    ----------
    model_output : the raw output from run_model().
    shape_in : the model input shape from setup_dpu_buffers().

    Returns
    -------
    heatmap : a 2D NumPy array you can show with plt.imshow(..., cmap='hot').

    Example
    -------
        heatmap = make_heatmap(output, shape_in)
    """
    heatmap = np.squeeze(np.max(model_output[0][:][:][:], axis=2))
    heatmap = cv2.resize(heatmap, (shape_in[2], shape_in[1]))
    return heatmap


def compare_poses(img1, img2, stride=15, show_comp=False):
    """
    Compare two pose heatmaps and return a match score (lower = better match).

    This slides one heatmap across the other and measures how different they
    are at each position, keeping the best (smallest) difference.

    Parameters
    ----------
    img1, img2 : two heatmap images to compare.
    stride : how many pixels to slide each step (default 15).
    show_comp : if True, show the comparison grid (for curiosity).

    Returns
    -------
    score : a number — smaller means the two poses are more alike.

    Example
    -------
        score = compare_poses(template, my_heatmap)
    """
    import matplotlib.pyplot as plt

    # Convert heatmaps to grayscale if they aren't already
    try:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    except Exception:
        pass
    try:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    except Exception:
        pass

    # Pad the first image so the second can slide fully across it
    xpad = int(img2.shape[1] / 2)
    ypad = int(img2.shape[0] / 2)
    canvas1 = np.zeros((img1.shape[0] + 2 * ypad, img1.shape[1] + 2 * xpad), dtype=img1.dtype)
    canvas1[ypad:img1.shape[0] + ypad, xpad:img1.shape[1] + xpad] = img1

    comp_x = int(img1.shape[1] / stride)
    comp_y = int(img1.shape[0] / stride)
    comparison_matrix = np.zeros((comp_x, comp_y), dtype=np.int32)

    for x in range(comp_x):
        for y in range(comp_y):
            canvas2 = np.zeros((canvas1.shape[0], canvas1.shape[1]), dtype=canvas1.dtype)
            canvas2[y * stride:img2.shape[0] + y * stride,
                    x * stride:img2.shape[1] + x * stride] = img2
            mse = ((canvas1 - canvas2) * 2).sum()
            comparison_matrix[x][y] = mse

    if show_comp:
        plt.imshow(comparison_matrix)
        plt.show()

    return comparison_matrix.min() / 1000000
