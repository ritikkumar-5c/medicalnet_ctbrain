"""Loss functions for the CT-brain classifier.

Mirrors ct_brain_classifier: weighted CE (default), focal, and a cost-sensitive
loss that penalizes under-calling pathology to 'normal' (the clinical priority).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CostSensitiveLoss(nn.Module):
    """Minimize expected misclassification cost E_j[ p_j * C[true, j] ] (+ a small
    CE term for stability). C has 0 on the diagonal and high cost for under-calling
    pathology to 'normal', so the model keeps probability mass off 'normal' for
    near_normal / abnormal cases."""
    def __init__(self, cost_matrix, weight=None, ce_lambda=0.3, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("cost", cost_matrix.float())
        self.weight = weight
        self.ce_lambda = ce_lambda
        self.ls = label_smoothing

    def forward(self, logits, target):
        p = torch.softmax(logits, dim=1)
        cost_rows = self.cost.to(logits.device)[target]          # (B, C)
        loss = (p * cost_rows).sum(dim=1).mean()
        if self.ce_lambda > 0:
            loss = loss + self.ce_lambda * F.cross_entropy(
                logits, target, weight=self.weight, label_smoothing=self.ls)
        return loss


def build_cost_matrix(num_classes, cost_miss_abnormal=5.0, cost_miss_near_normal=3.0):
    """C[true, pred]: 0 diagonal, 1 generic error, higher for under-calling to normal (idx 0)."""
    C = torch.ones(num_classes, num_classes) - torch.eye(num_classes)
    if num_classes >= 3:                 # 0=normal, 1=near_normal, 2=abnormal
        C[2, 0] = float(cost_miss_abnormal)
        C[1, 0] = float(cost_miss_near_normal)
    elif num_classes == 2:
        C[1, 0] = float(cost_miss_abnormal)
    return C


def build_loss(tc, num_classes, weight=None):
    """tc = cfg['train'] dict. Returns the criterion module."""
    loss = tc.get("loss", "weighted_ce")
    ls = tc.get("label_smoothing", 0.0)
    if loss == "focal":
        return FocalLoss(weight=weight, gamma=tc.get("focal_gamma", 2.0), label_smoothing=ls)
    if loss == "cost_sensitive":
        C = build_cost_matrix(num_classes,
                              tc.get("cost_miss_abnormal", 5.0),
                              tc.get("cost_miss_near_normal", 3.0))
        return CostSensitiveLoss(C, weight=weight,
                                 ce_lambda=tc.get("cost_ce_lambda", 0.3), label_smoothing=ls)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
