import torch
import torch.nn as nn


class SharedSupermaskTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        max_seq_len=512,
        prune_rate=0.5,
        dropout=0.1
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Shared layer: only one!
        self.shared_block = SupermaskTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            prune_rate=prune_rate,
            dropout=dropout
        )

        # Create L separate sets of scores (Supermasks) for reuse
        self.masks = nn.ModuleList([
            copy.deepcopy(self.shared_block)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        x = self.embed(x) + self.pos_embed[:, :T]

        # Forward through shared weights, different masks
        for i, block in enumerate(self.masks):
            # Point all layers to shared weights
            self._share_weights(block, self.shared_block)
            x = block(x)

        x = self.norm(x)
        return self.output_head(x)

    def _share_weights(self, block, shared_block):
        """
        Make all weights in `block` point to those in `shared_block`.
        Each block has its own scores (mask), but shared W.
        """
        for (name, module), (sname, smodule) in zip(block.named_modules(), shared_block.named_modules()):
            if isinstance(module, SupermaskLinear) and isinstance(smodule, SupermaskLinear):
                module.weight = smodule.weight  # Shared W
                module.training_or_inference = smodule.training_or_inference
                module.current_mask = None  # force recompute







'''
model = SharedSupermaskTransformer(
    vocab_size=10000,
    embed_dim=512,
    num_heads=8,
    ff_dim=2048,
    num_layers=6,          # Simulates a 6-layer Transformer
    prune_rate=0.5,
    dropout=0.1
)

dummy_input = torch.randint(0, 10000, (32, 64))  # (batch_size, seq_len)
logits = model(dummy_input)  # Output: (32, 64, 10000)

'''
