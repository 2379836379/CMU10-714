import os
import numpy as np
from needle import Tensor


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        ids = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                words = line.strip().split() + ['<eos>']
                ids.extend(self.dictionary.add_word(word) for word in words)
        return ids


def batchify(data, batch_size, device, dtype):
    nbatch = len(data) // batch_size
    data = np.array(data[:nbatch * batch_size], dtype=np.float32)
    return data.reshape((batch_size, nbatch)).T


def get_batch(batches, i, bptt, device=None, dtype=None):
    seq_len = min(bptt, batches.shape[0] - 1 - i)
    data = Tensor(batches[i:i + seq_len], device=device, dtype=dtype, requires_grad=False)
    target = Tensor(batches[i + 1:i + 1 + seq_len].reshape((-1,)), device=device, dtype=dtype, requires_grad=False)
    return data, target
