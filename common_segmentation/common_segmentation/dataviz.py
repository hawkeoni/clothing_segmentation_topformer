import cv2
import numpy as np


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


def make_colored_mask(mask):
    colored_mask = np.zeros(list(mask.shape) + [3], dtype=np.uint8)
    colored_mask[mask == 1] = [200, 0, 0]
    colored_mask[mask == 2] = [0, 200, 0]
    colored_mask[mask == 3] = [0, 0, 200]
    return colored_mask


def add_mask(image: np.ndarray, mask: np.ndarray):
    return cv2.addWeighted(image, 1, mask, 0.8, 0)

