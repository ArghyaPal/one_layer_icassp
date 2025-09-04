import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SupermaskMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, prune_rate=0.5, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.prune_rate = prune_rate

        self.q_proj = SupermaskLinear(embed_dim, embed_dim, prune_rate)
        self.k_proj = SupermaskLinear(embed_dim, embed_dim, prune_rate)
        self.v_proj = SupermaskLinear(embed_dim, embed_dim, prune_rate)
        self.out_proj = SupermaskLinear(embed_dim, embed_dim, prune_rate)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, embed_dim]
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, T, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)


class SupermaskTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, prune_rate=0.5, dropout=0.1):
        super().__init__()

        self.attn = SupermaskMultiHeadAttention(embed_dim, num_heads, prune_rate, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            SupermaskLinear(embed_dim, ff_dim, prune_rate),
            nn.ReLU(),
            SupermaskLinear(ff_dim, embed_dim, prune_rate),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention + Add & Norm
        attn_out = self.attn(x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Feedforward + Add & Norm
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x
