from torch.utils.data import DataLoader

# Reload dataset iterator
train_iter, valid_iter = IWSLT2016(split=('train', 'valid'), language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

# Wrap in DataLoader
valid_dataset = list(valid_iter)
valid_loader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_batch)

# Evaluate
compute_bleu(model, valid_loader, vocab_transform[TGT_LANGUAGE])
