import torch
from torch.utils.data import DataLoader
from dataloader import get_tokenizers, build_vocabs, get_iwslt_dataset, collate_batch
from bleu import compute_bleu
from SharedSupermaskTransformer import SharedSupermaskTransformer

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='Fraction of weights to keep')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--dim_model', type=int, default=512, help='Embedding/hidden size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device

    # 1. Tokenizers and Vocabs
    token_transform, vocab_transform = get_tokenizers()
    vocab_transform = build_vocabs(token_transform)

    # 2. Dataset + Dataloader
    train_data, valid_data = get_iwslt_dataset()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, token_transform, vocab_transform))
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, token_transform, vocab_transform))

    # 3. Model
    model = SharedSupermaskTransformer(
        vocab_size=len(vocab_transform['de']),
        tgt_vocab_size=len(vocab_transform['en']),
        dim_model=args.dim_model,
        num_layers=args.num_layers,
        prune_ratio=args.prune_ratio
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4. Training loop
    print(f"Starting training for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1),
                ignore_index=0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}")

    # 5. BLEU evaluation
    print("Evaluating BLEU score...")
    bleu = compute_bleu(model, valid_loader, vocab_transform['en'])
    print(f"\nFinal BLEU score: {bleu:.2f}")

if __name__ == "__main__":
    main()
