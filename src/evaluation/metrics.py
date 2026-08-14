"""Lesion-level box parsing, IoU, matching, and metric helpers."""

import json

import numpy as np


def boxes(value):
    x = json.loads(value) if isinstance(value, str) else value
    return np.asarray(x, dtype=float).reshape(-1, 4)


def scores(value):
    x = json.loads(value) if isinstance(value, str) else value
    return np.asarray(x, dtype=float)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(br - tl, 0), axis=2)
    aa = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)[:, None]
    bb = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)[None, :]
    return np.divide(
        inter, aa + bb - inter, out=np.zeros_like(inter), where=(aa + bb - inter) > 0
    )


def match(pred, gt, threshold=0.5):
    matrix = iou_matrix(pred, gt)
    candidates = []
    for pi in range(len(pred)):
        for gi in range(len(gt)):
            if matrix[pi, gi] >= threshold:
                candidates.append((matrix[pi, gi], pi, gi))
    used_p = set()
    used_g = set()
    pairs = []
    for value, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            pairs.append((pi, gi, float(value)))
    return pairs, len(pred) - len(pairs), len(gt) - len(pairs)


def safe(a, b):
    return float(a / b) if b else 0.0
