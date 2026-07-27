"""
vision.py — shared image and camera helpers.

These wrap the most common OpenCV / matplotlib patterns that show up in almost
every notebook: reading an image file, showing an image, showing two images
side-by-side, and grabbing a picture from the webcam. Used by PYNQ 201, 501,
503, and any notebook that works with pictures.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(path, to_rgb=True):
    """
    Read an image file from disk.

    OpenCV loads images in "BGR" color order, but matplotlib expects "RGB", so
    by default we flip it to RGB so it looks right when you show it.

    Parameters
    ----------
    path : the image file path, e.g. "img/mosaic.jpg".
    to_rgb : flip BGR->RGB so colors look correct (default True).

    Returns
    -------
    image : the image as a NumPy array.

    Example
    -------
        img = read_image("img/mosaic.jpg")
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(path))
    if to_rgb and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_image(image, title=None, figsize=(10, 10), cmap=None):
    """
    Show a single image in the notebook (no axis ticks).

    Example
    -------
        show_image(img, title="My picture")
    """
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.show()


def plot_images(original_image, processed_image, figsize=(15, 15),
                titles=("Original", "Processed")):
    """
    Show two images side-by-side ("before" and "after").

    This is the same helper used all over the Computer Vision notebook — great
    for seeing what an image filter did.

    Example
    -------
        plot_images(img, gray_img)
    """
    plt.figure(figsize=figsize)
    plt.subplot(121), plt.imshow(original_image), plt.title(titles[0])
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(processed_image), plt.title(titles[1])
    plt.xticks([]), plt.yticks([])
    plt.show()


def open_camera(index=-1, width=None, height=None, no_buffer=True):
    """
    Open the webcam so you can take pictures with it.

    Parameters
    ----------
    index : which camera to use (-1 or 0 usually works).
    width, height : optional frame size to request.
    no_buffer : throw away old buffered frames so you get the latest one.

    Returns
    -------
    camera : the camera object. Use capture_frame(camera) to take a picture,
             and remember to call camera.release() when you're done.

    Example
    -------
        camera = open_camera()
        frame = capture_frame(camera)
        camera.release()
    """
    camera = cv2.VideoCapture(index)
    if no_buffer:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width is not None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def capture_frame(camera, to_rgb=False):
    """
    Take a single picture from an open camera.

    Parameters
    ----------
    camera : a camera from open_camera().
    to_rgb : flip BGR->RGB so colors look correct when shown with matplotlib.
             Leave False if you're feeding the frame straight into a model.

    Returns
    -------
    frame : the captured image, or None if the camera failed.

    Example
    -------
        frame = capture_frame(camera, to_rgb=True)
    """
    ok, frame = camera.read()
    if not ok:
        return None
    if to_rgb and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def list_images(image_folder, extensions=("JPEG", "jpg", "jpeg", "png")):
    """
    Return a sorted list of image filenames in a folder.

    Returns
    -------
    image_folder, filenames

    Example
    -------
        folder, files = list_images("img")
    """
    exts = tuple(e.lower() for e in extensions)
    files = sorted(
        f for f in os.listdir(image_folder)
        if f.rsplit(".", 1)[-1].lower() in exts
    )
    return image_folder, files
