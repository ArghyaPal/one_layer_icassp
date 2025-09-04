import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class TopKMask(Function):
    """
    Differentiable Top-k masking using the Straight-Through Estimator (STE)
    """
    @staticmethod
    def forward(ctx, scores, k):
        # scores: importance scores (learnable)
        # k: fraction of weights to keep
        flat_scores = scores.view(-1)
        num_params = flat_scores.size(0)
        kth = int((1 - k) * num_params)

        # Get the threshold value for top-k%
        threshold = flat_scores.kthvalue(kth).values
        mask = (scores >= threshold).float()
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: just pass gradients as-is
        return grad_output, None


class SupermaskLinear(nn.Module):
    def __init__(self, in_features, out_features, prune_rate=0.5, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prune_rate = prune_rate

        # Fixed random weight W
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False  # Freeze W

        # Learnable importance scores S
        self.scores = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.scores, mean=0.0, std=0.1)

        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        mask = TopKMask.apply(self.scores, self.prune_rate)
        w_pruned = self.weight * mask
        return F.linear(x, w_pruned, self.bias)
