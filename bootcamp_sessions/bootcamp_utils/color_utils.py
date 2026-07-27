"""
color_utils.py — shared helpers for named classes and their colors.

Used by any notebook that has a list of class names (like "cat", "dog", "car")
and wants a distinct color for each one — for example when drawing labeled
boxes in the Object Detection session (PYNQ 301).
"""
import random
import colorsys


def load_classes(classes_path):
    """
    Read class names from a text file, one name per line.

    Example
    -------
        class_names = load_classes("img/voc_classes.txt")
    """
    with open(classes_path) as f:
        return [c.strip() for c in f.readlines()]


def make_colors(class_names, seed=0):
    """
    Make a unique, nicely-spread color for each class name.

    Returns
    -------
    list of (R, G, B) int tuples — one color per class.

    Example
    -------
        colors = make_colors(class_names)
    """
    n = len(class_names)
    hsv_tuples = [(x / n, 1.0, 1.0) for x in range(n)]
    colors = [colorsys.hsv_to_rgb(*h) for h in hsv_tuples]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    random.seed(seed)
    random.shuffle(colors)
    random.seed(None)
    return colors
