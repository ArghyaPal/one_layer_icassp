import torch
import torch.nn as nn
from SupermaskTransformerBlock import SupermaskTransformerBlock


class SharedSupermaskTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout=0.1,
        max_len=512,
        prune_ratio=0.5,
        mask_init="standard",
        prune_method="topk"
    ):
        super(SharedSupermaskTransformer, self).__init__()

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.embed_scale = embed_dim ** 0.5
        self.max_len = max_len

        self.num_layers = num_layers

        # === Shared randomly initialized Transformer block === #
        self.shared_block = SupermaskTransformerBlock(
            embed_dim,
            num_heads,
            ff_dim,
            dropout,
            prune_ratio=prune_ratio,
            mask_init=mask_init,
            prune_method=prune_method
        )

        # === Learnable supermask scores per layer === #
        self.mask_scores_per_layer = nn.ModuleList([
            self.shared_block.init_mask_scores()
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        """
        bsz, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_len)

        x = self.embed_scale * self.embed_tokens(input_ids) + self.pos_embedding(position_ids)
        x = self.dropout(x)

        for l in range(self.num_layers):
            x = self.shared_block(
                x,
                attention_mask=attention_mask,
                mask_scores=self.mask_scores_per_layer[l]
            )

        x = self.output_proj(x)
        return x

    def change_mode(self, mode="train"):
        """
        Switch between 'train' and 'eval' mode for deterministic masking
        """
        assert mode in ["train", "eval"], f"Invalid mode: {mode}"
        self.shared_block.change_mode(mode)
        self.train(mode == "train")

    def get_all_masks(self):
        return [self.shared_block.get_mask(scores) for scores in self.mask_scores_per_layer]
