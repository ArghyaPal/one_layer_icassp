from torchtext.datasets import IWSLT2016
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
    TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm')
}

def yield_tokens(data_iter, language):
    for src, tgt in data_iter:
        yield token_transform[language](src if language == SRC_LANGUAGE else tgt)

train_iter = IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

vocab_transform = {}
for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[lang] = build_vocab_from_iterator(
        yield_tokens(train_iter, lang), specials=["<pad>", "<unk>", "<bos>", "<eos>"], min_freq=2)
    vocab_transform[lang].set_default_index(vocab_transform[lang]["<unk>"])


####################
from torch.nn.utils.rnn import pad_sequence

def sequential_transforms(*transforms):
    def func(txt):
        for transform in transforms:
            txt = transform(txt)
        return txt
    return func

def tensor_transform(token_ids):
    return torch.cat([torch.tensor([bos_idx]), torch.tensor(token_ids), torch.tensor([eos_idx])])

text_transform = {}
for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[lang] = sequential_transforms(
        token_transform[lang],
        vocab_transform[lang],
        tensor_transform
    )

pad_idx = vocab_transform[SRC_LANGUAGE]['<pad>']
bos_idx = vocab_transform[SRC_LANGUAGE]['<bos>']
eos_idx = vocab_transform[SRC_LANGUAGE]['<eos>']

def collate_batch(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))
    src_batch = pad_sequence(src_batch, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
    return src_batch.T, tgt_batch.T
