import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from supermask_linear import SupermaskLinear


class SupermaskMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, prune_rate=0.5, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = SupermaskLinear(embed_dim, embed_dim, prune_ratio=prune_rate)
        self.k_proj = SupermaskLinear(embed_dim, embed_dim, prune_ratio=prune_rate)
        self.v_proj = SupermaskLinear(embed_dim, embed_dim, prune_ratio=prune_rate)
        self.out_proj = SupermaskLinear(embed_dim, embed_dim, prune_ratio=prune_rate)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)


class SupermaskTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, prune_ratio=0.5, dropout=0.1,
                 mask_init='standard', prune_method='topk'):
        super().__init__()

        self.attn = SupermaskMultiHeadAttention(embed_dim, num_heads, prune_ratio, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            SupermaskLinear(embed_dim, ff_dim, prune_ratio, mask_init=mask_init, prune_method=prune_method),
            nn.ReLU(),
            SupermaskLinear(ff_dim, embed_dim, prune_ratio, mask_init=mask_init, prune_method=prune_method),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, mask_scores=None):
        if mask_scores is not None:
            self.attn.q_proj.set_scores(mask_scores['attn_q'])
            self.attn.k_proj.set_scores(mask_scores['attn_k'])
            self.attn.v_proj.set_scores(mask_scores['attn_v'])
            self.attn.out_proj.set_scores(mask_scores['attn_out'])
            self.ff[0].set_scores(mask_scores['ff_1'])
            self.ff[2].set_scores(mask_scores['ff_2'])

        attn_out = self.attn(x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x

    def change_mode(self, mode="train"):
        self.attn.q_proj.change_mode(mode)
        self.attn.k_proj.change_mode(mode)
        self.attn.v_proj.change_mode(mode)
        self.attn.out_proj.change_mode(mode)
        self.ff[0].change_mode(mode)
        self.ff[2].change_mode(mode)

    def init_mask_scores(self):
        return nn.ModuleDict({
            'attn_q': self.attn.q_proj.init_scores(),
            'attn_k': self.attn.k_proj.init_scores(),
            'attn_v': self.attn.v_proj.init_scores(),
            'attn_out': self.attn.out_proj.init_scores(),
            'ff_1': self.ff[0].init_scores(),
            'ff_2': self.ff[2].init_scores(),
        })
