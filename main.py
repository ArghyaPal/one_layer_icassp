import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from dataloader import get_dataloaders, SRC_LANGUAGE, TGT_LANGUAGE
from SharedSupermaskTransformer import SharedSupermaskTransformer
from bleu import compute_bleu


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch['src'].to(device)
        targets = batch['tgt'].to(device)

        optimizer.zero_grad()
        logits = model(inputs, targets)  # [B, T, V]

        loss = criterion(
            logits.view(-1, logits.size(-1)),  # [B*T, V]
            targets.view(-1)                   # [B*T]
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, tgt_vocab, pad_idx, bos_idx, eos_idx, device):
    model.eval()
    with torch.no_grad():
        bleu = compute_bleu(model, dataloader, tgt_vocab, pad_idx, bos_idx, eos_idx)
    return bleu


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data + vocab
    train_loader, valid_loader, src_vocab, tgt_vocab = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    pad_idx = src_vocab["<pad>"]
    bos_idx = src_vocab["<bos>"]
    eos_idx = src_vocab["<eos>"]

    # Model
    model = SharedSupermaskTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        prune_ratio=args.prune_ratio,
        init_type=args.init_type
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        bleu_score = evaluate(model, valid_loader, tgt_vocab, pad_idx, bos_idx, eos_idx, device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | BLEU: {bleu_score:.2f}")

    print("✅ Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iwslt2017', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--ffn_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--prune_ratio', type=float, default=0.5)
    parser.add_argument('--init_type', type=str, default='kaiming_uniform')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()

    main(args)
