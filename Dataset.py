import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.datasets import IWSLT2016
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
    TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm')
}

SPECIAL_SYMBOLS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Lazy setup for vocab
def yield_tokens(data_iter, language):
    for src, tgt in data_iter:
        yield token_transform[language](src if language == SRC_LANGUAGE else tgt)


def build_vocabs(min_freq=2):
    train_iter = IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    vocab_transform = {}

    for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab = build_vocab_from_iterator(
            yield_tokens(train_iter, lang),
            specials=SPECIAL_SYMBOLS,
            min_freq=min_freq
        )
        vocab.set_default_index(vocab["<unk>"])
        vocab_transform[lang] = vocab

    return vocab_transform


def sequential_transforms(*transforms):
    def func(txt):
        for transform in transforms:
            txt = transform(txt)
        return txt
    return func


def tensor_transform(token_ids, bos_idx, eos_idx):
    return torch.tensor([bos_idx] + token_ids + [eos_idx])


def build_text_transform(vocab_transform):
    text_transform = {}
    for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
        bos_idx = vocab_transform[lang]["<bos>"]
        eos_idx = vocab_transform[lang]["<eos>"]
        text_transform[lang] = sequential_transforms(
            token_transform[lang],
            vocab_transform[lang],
            lambda x: tensor_transform(x, bos_idx, eos_idx)
        )
    return text_transform


def collate_batch_fn(vocab_transform, text_transform):
    pad_idx = vocab_transform[SRC_LANGUAGE]["<pad>"]

    def collate_batch(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_tensor = text_transform[SRC_LANGUAGE](src_sample)
            tgt_tensor = text_transform[TGT_LANGUAGE](tgt_sample)
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)

        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)

        return {"src": src_batch.T, "tgt": tgt_batch.T}

    return collate_batch


def get_dataloaders(dataset_name="iwslt2016", batch_size=64, max_len=128):
    assert dataset_name.lower() == "iwslt2016", "Only IWSLT2016 is supported right now."

    vocab_transform = build_vocabs()
    text_transform = build_text_transform(vocab_transform)
    collate_fn = collate_batch_fn(vocab_transform, text_transform)

    train_iter = IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    valid_iter = IWSLT2016(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

    train_loader = DataLoader(
        train_iter,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_iter,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    src_vocab_size = len(vocab_transform[SRC_LANGUAGE])
    tgt_vocab_size = len(vocab_transform[TGT_LANGUAGE])

    return train_loader, valid_loader, src_vocab_size, tgt_vocab_size
