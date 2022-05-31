from typing import Dict, List, Union

import numpy as np



def calculate_iou(true_mask: np.ndarray, pred_mask: np.ndarray):
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    return intersection / (union + 1e-9)


def calculate_multiclass_iou(true_labels: np.ndarray, pred_labels: np.ndarray, classes: Union[int, List[str]]) -> Dict[str, float]:
    if isinstance(classes, int):
        N = classes
        classnames = {i: i for i in range(N)}
    else:
        N = len(classes)
        classnames = classes

    res = {}
    # skipping 0 as background
    for i in range(1, N):
        res[classnames[i]] = calculate_iou(true_labels == i, pred_labels == i)
    return res
