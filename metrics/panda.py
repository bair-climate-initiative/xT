import numpy as np
from collections import Counter
from typing import List
import torch
from torch import Tensor


gleason_to_isup = {
    # majority + minority -> ISUP
    '0+0': 0,
    "3+3": 1,
    "3+4": 2,
    "4+3": 3,
    "4+4": 4,
    "3+5": 4,
    "5+3": 4,
    "4+5": 5,
    "5+4": 5,
    "5+5": 5,
}

def score_image(gleason_scores: List[str]):
    return Counter(gleason_scores).most_common(3)
    

def quadratic_kappa_coefficient(output: Tensor, target: Tensor):
    """
    https://www.kaggle.com/mawanda/qwk-metric-in-pytorch
    """
    output, target = output.type(torch.float32), target.type(torch.float32)
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (
        n_classes - 1
    )
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum()  # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK