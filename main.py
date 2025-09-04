import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Dummy training
# Config
vocab_size = 10000
embed_dim = 256
num_heads = 4
ff_dim = 1024
num_layers = 6
batch_size = 32
seq_len = 32
num_samples = 10000
prune_rate = 0.5

# Dataset + Model
dataset = DummyTranslationDataset(vocab_size, seq_len, num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SharedSupermaskTransformer(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_layers=num_layers,
    prune_rate=prune_rate
)

# Optimizer: only update scores
mask_params = [p for n, p in model.named_parameters() if "scores" in n]
optimizer = optim.Adam(mask_params, lr=1e-3)

# Train
train_supermask_transformer(model, dataloader, optimizer, num_epochs=10)
