import sacrebleu
import torch


def generate(model, src_batch, bos_idx, eos_idx, max_len=50):
    model.eval()
    device = next(model.parameters()).device
    out_tokens = []

    with torch.no_grad():
        for src in src_batch:
            src = src.unsqueeze(0).to(device)
            ys = torch.tensor([[bos_idx]], device=device)

            for _ in range(max_len):
                out = model(src, ys)  # [1, seq_len, vocab_size]
                next_word = out[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_word], dim=1)
                if next_word.item() == eos_idx:
                    break

            out_tokens.append(ys.squeeze(0).tolist())

    return out_tokens  # list of token IDs


def detokenize(tokens_batch, vocab, pad_idx, bos_idx, eos_idx):
    itos = vocab.get_itos()
    sentences = []
    for tokens in tokens_batch:
        words = [
            itos[token]
            for token in tokens
            if token not in (pad_idx, bos_idx, eos_idx)
        ]
        sentences.append(" ".join(words))
    return sentences


def compute_bleu(model, data_loader, tgt_vocab, pad_idx, bos_idx, eos_idx):
    references = []
    hypotheses = []

    model.eval()
    for batch in data_loader:
        src_batch = batch["src"]
        tgt_batch = batch["tgt"]

        preds = generate(model, src_batch, bos_idx, eos_idx)
        hyps = detokenize(preds, tgt_vocab, pad_idx, bos_idx, eos_idx)
        refs = detokenize(tgt_batch.tolist(), tgt_vocab, pad_idx, bos_idx, eos_idx)

        references.extend([[ref] for ref in refs])
        hypotheses.extend(hyps)

    bleu = sacrebleu.corpus_bleu(hypotheses, list(zip(*references)))
    print(f"BLEU score: {bleu.score:.2f}")
    return bleu.score
