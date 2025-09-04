import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class TopKMask(Function):
    """
    Differentiable Top-k masking using the Straight-Through Estimator (STE)
    """
    @staticmethod
    def forward(ctx, scores, k):
        flat_scores = scores.view(-1)
        num_params = flat_scores.size(0)
        kth = int((1 - k) * num_params)

        threshold = flat_scores.kthvalue(kth).values
        mask = (scores >= threshold).float()
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class SupermaskLinear(nn.Module):
    def __init__(self, in_features, out_features, prune_ratio=0.5, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prune_ratio = prune_ratio

        # Random fixed weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        # Learnable importance scores
        self.scores = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.scores, mean=0.0, std=0.1)

        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Cache mask if in eval mode
        self.cached_mask = None
        self.training_or_inference = "train"

    def forward(self, x):
        if self.training_or_inference == "train":
            self.cached_mask = None
            mask = TopKMask.apply(self.scores, self.prune_ratio)
        else:
            if self.cached_mask is None:
                with torch.no_grad():
                    mask = TopKMask.apply(self.scores, self.prune_ratio)
                    self.cached_mask = mask
            mask = self.cached_mask

        w_pruned = self.weight * mask
        return F.linear(x, w_pruned, self.bias)

    def set_scores(self, new_scores):
        """
        Allow external replacement of scores (e.g., shared weight + unique masks)
        """
        with torch.no_grad():
            self.scores.copy_(new_scores)

    def init_scores(self):
        """
        Return a new, independent copy of current scores (used for virtual supermasks)
        """
        return nn.Parameter(self.scores.data.clone(), requires_grad=True)

    def change_mode(self, mode="train"):
        """
        Switch between training or inference mode (to cache masks for eval)
        """
        assert mode in ["train", "eval"]
        self.training_or_inference = mode
