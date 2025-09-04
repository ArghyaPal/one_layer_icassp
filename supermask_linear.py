import torch
import torch.nn as nn
import torch.nn.functional as F


class SupermaskLinear(nn.Module):
    def __init__(self, in_features, out_features, pruning_ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pruning_ratio = pruning_ratio

        # Fixed random weights (not updated)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # Kaiming Uniform as in the paper

        # Learnable importance scores
        self.scores = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.scores, a=5**0.5)

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def get_mask(self):
        # Top-k masking: keep top (k%) of scores
        num_elements = self.scores.numel()
        k = int(self.pruning_ratio * num_elements)

        # Flatten scores and find top-k indices
        flat_scores = self.scores.view(-1)
        topk = torch.topk(flat_scores, k)
        threshold = topk.values[-1]  # smallest top-k value

        # Create binary mask
        mask = (self.scores >= threshold).float()
        return mask

    def forward(self, x):
        # Compute mask with STE
        mask = self.get_mask()
        masked_weight = self.weight * mask

        # Forward with masked weights
        return F.linear(x, masked_weight, self.bias)
