import numpy as np
import torch
from torchmetrics.functional import bleu_score


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )

    return (matrix[size_x - 1, size_y - 1])


def word_levenshtein(preds, target):
    # Word-level Levenshtein Distance
    # adopted from: https://github.com/mciniselli/T5_Replication_Package/blob/main/Score/metrics.py
    """
        Examples:
            >>> target = "this is the reference".split()
            >>> preds = "this is the prediction".split()
            >>> word_levenshtein(target=target, preds=preds)
            tensor(0.2500)
    """

    levenshtein_distance = levenshtein(preds, target)

    prediction_length = len(preds)
    target_length = len(target)
    lev_distance_perc = 0

    if prediction_length > target_length:
        lev_distance_perc = levenshtein_distance / prediction_length
    else:
        if target_length == 0:
            lev_distance_perc = -1
            print("target_length = 0")
        else:
            lev_distance_perc = levenshtein_distance / target_length

    return torch.scalar_tensor(lev_distance_perc)


def stack_scores(scores):
    stacked_list = {}
    for out in scores:
        for key, val in out.items():
            if key in stacked_list:
                stacked_list[key].append(val)
            else:
                stacked_list[key] = [val]

    _template = {}
    for key, val in stacked_list.items():
        _template[key] = torch.stack(val).mean().item()

    return _template
