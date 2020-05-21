import math
import os
import random
from collections import Counter, defaultdict

import torch

PAD_WORD = '<pad>' # 0
UNK_WORD = '<unk>' # 1
BOS_WORD = '<s>'   # 2
EOS_WORD = '</s>'  # 3

class Vocab(object):
    def __init__(self, lang=None, config=None):
        self.specials = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
        self.counter = Counter()
        self.stoi = {}
        self.itos = {}
        self.lang = lang
        self.weights = None
        self.min_freq = config.min_freq

    def load_vocab(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self.stoi and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file."
                            .format(word)
                    )
                self.counter[word] = count
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

        if self.min_freq > 1:
            self.counter = {w:i for w, i in filter(
                lambda x:x[1] >= self.min_freq, self.counter.items())}

        self.vocab_size = 0
        for w in self.specials:
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1

        for w in self.counter.keys():
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1

        self.itos = {i: w for w, i in self.stoi.items()}

    def __len__(self):
        return self.vocab_size


class DataSet(list):
    def __init__(self, *args, config=None, is_train=True, dataset="train"):
        self.config = config
        self.is_train = is_train
        self.src_lang = config.src_lang
        self.trg_lang = config.trg_lang
        self.dataset = dataset
        self.data_path = (
            os.path.join(self.config.data_path, dataset + "." + self.src_lang),
            os.path.join(self.config.data_path, dataset + "." + self.trg_lang)
        )
        super(DataSet, self).__init__(*args)

    def read(self):
        with open(self.data_path[0], "r") as fin_src, \
                open(self.data_path[1], "r") as fin_trg:
            for line1, line2 in zip(fin_src, fin_trg):
                src, trg = line1.rstrip("\r\n"), line2.rstrip("\r\n")
                src = src.split()
                trg = trg.split()
                if self.is_train:
                    if len(src) <= self.config.max_seq_len and \
                            len(trg) <= self.config.max_seq_len:
                        self.append((src, trg))
                else:
                    self.append((src, trg))
        fin_src.close()
        fin_trg.close()

    def _numericalize(self, words, stoi):
        return [1 if x not in stoi else stoi[x] for x in words]

    def numericalize(self, src_w2id, trg_w2id):
        for i, example in enumerate(self):
            x, y = example
            x = self._numericalize(x, src_w2id)
            y = self._numericalize(y, trg_w2id)
            self[i] = (x, y)


class DataBatchIterator(object):
    def __init__(self, config, dataset="train",
                 is_train=True, batch_size=64,
                 shuffle=False, sample=False,
                 sort_in_batch=True):
        self.config = config
        self.examples = DataSet(config=config, is_train=is_train, dataset=dataset)
        self.src_vocab = Vocab(lang=config.src_lang, config=config)
        self.trg_vocab = Vocab(lang=config.trg_lang, config=config)
        self.is_train = (dataset == "train")
        self.max_seq_len = config.max_seq_len
        self.sort_in_batch = sort_in_batch
        self.is_shuffle = shuffle
        self.is_sample = sample
        self.batch_size = batch_size
        self.num_batches = 0

    def set_vocab(self, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def load(self, vocab_cache=None):
        if not vocab_cache and self.is_train:
            self.examples.read()
            self.src_vocab.make_vocab([x[0] for x in self.examples])
            self.trg_vocab.make_vocab([x[1] for x in self.examples])
            self.examples.numericalize(
                src_w2id=self.src_vocab.stoi,
                trg_w2id=self.trg_vocab.stoi)

            # self.src_vocab.load_pretrained_embedding(
            #     self.config.embed_path+".%s"%self.config.source_lang, self.config.embed_dim)
            # self.trg_vocab.load_pretrained_embedding(
            #     self.config.embed_path+".%s"%self.config.target_lang, self.config.embed_dim)

        if not self.is_train:
            self.examples.read()
            assert len(self.src_vocab) > 0
            self.examples.numericalize(
                src_w2id=self.src_vocab.stoi,
                trg_w2id=self.trg_vocab.stoi)
            # self.src_vocab.load_pretrained_embedding(
            #     self.config.embed_path+".%s"%self.config.source_lang, self.config.embed_dim)
            # self.trg_vocab.load_pretrained_embedding(
            #     self.config.embed_path+".%s"%self.config.target_lang, self.config.embed_dim)
        self.num_batches = math.ceil(len(self.examples) / self.batch_size)

    def _pad(self, sentence, max_L, w2id, add_bos=False, add_eos=False):
        if add_bos:
            sentence = [w2id[BOS_WORD]] + sentence
        if add_eos:
            sentence = sentence + [w2id[EOS_WORD]]
        if len(sentence) < max_L:
            sentence = sentence + [w2id[PAD_WORD]] * (max_L - len(sentence))
        return [x for x in sentence]

    def pad_seq_pair(self, samples):
        if self.sort_in_batch:
            samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
        pairs = [pair for pair in samples]

        src_Ls = [len(pair[0]) + 2 for pair in pairs]
        trg_Ls = [len(pair[1]) + 2 for pair in pairs]

        max_trg_Ls = max(trg_Ls)
        max_src_Ls = max(src_Ls)
        src = [self._pad(
            src, max_src_Ls, self.src_vocab.stoi, add_bos=True, add_eos=True) for src, _ in pairs]
        trg = [self._pad(
            trg, max_trg_Ls, self.trg_vocab.stoi, add_bos=True, add_eos=True) for _, trg in pairs]

        batch = Batch()
        batch.src = torch.LongTensor(src).transpose(0, 1).cuda()
        batch.trg = torch.LongTensor(trg).transpose(0, 1).cuda()

        batch.src_Ls = torch.LongTensor(src_Ls).cuda()
        batch.trg_Ls = torch.LongTensor(trg_Ls).cuda()
        return batch

    def __iter__(self):
        if self.is_shuffle:
            random.shuffle(self.examples)
        total_num = len(self.examples)
        for i in range(self.num_batches):
            if self.is_sample:
                samples = random.sample(self.examples, self.batch_size)
            else:
                samples = self.examples[i * self.batch_size: \
                                        min(total_num, self.batch_size * (i + 1))]
            yield self.pad_seq_pair(samples)


class Batch(object):
    def __init__(self):
        self.src = None
        self.trg = None
        self.src_Ls = None
        self.trg_Ls = None

    def __len__(self):
        return self.src_Ls.size(0)

    @property
    def batch_size(self):
        return self.src_Ls.size(0)