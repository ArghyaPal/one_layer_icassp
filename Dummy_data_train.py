import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



# Dummy Dataset for Testing
class DummyTranslationDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(1, self.vocab_size, (self.seq_len,))
        y = torch.roll(x, shifts=-1)  # Shifted version as target
        return x, y


# Loss
criterion = nn.CrossEntropyLoss()


# Training Loop
def train_supermask_transformer(model, dataloader, optimizer, num_epochs=5, device="cuda"):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # Output: (B, T, vocab_size)

            # Flatten to apply cross-entropy
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {total_loss / len(dataloader):.4f}")


