import sacrebleu

def generate(model, src_batch, max_len=50):
    model.eval()
    with torch.no_grad():
        out_tokens = []
        for src in src_batch:
            src = src.unsqueeze(0).to(next(model.parameters()).device)
            ys = torch.tensor([[bos_idx]], device=src.device)
            for _ in range(max_len):
                out = model(src, ys)
                next_word = out[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_word], dim=1)
                if next_word.item() == eos_idx:
                    break
            out_tokens.append(ys.squeeze(0).tolist())
        return out_tokens

def detokenize(tokens, vocab):
    itos = vocab.get_itos()
    return [" ".join([itos[token] for token in tok if token not in [pad_idx, bos_idx, eos_idx]]) for tok in tokens]

def compute_bleu(model, data_loader, vocab_tgt):
    references = []
    hypotheses = []
    for src_batch, tgt_batch in data_loader:
        preds = generate(model, src_batch)
        hyps = detokenize(preds, vocab_tgt)
        refs = detokenize(tgt_batch.tolist(), vocab_tgt)
        references.extend([[ref] for ref in refs])
        hypotheses.extend(hyps)
    bleu = sacrebleu.corpus_bleu(hypotheses, list(zip(*references)))
    print(f"BLEU score: {bleu.score:.2f}")
    return bleu.score
