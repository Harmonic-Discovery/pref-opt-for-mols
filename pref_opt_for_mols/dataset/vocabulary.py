"""
This file defines Vocabulary objects which are used to tokenize Smiles

The base class Vocabulary() is taken from Facebook's Fairseq package (https://github.com/facebookresearch/fairseq)
"""

import os
from collections import Counter

import torch


class Vocabulary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(self, pad="<pad>", eos="</s>", unk="<unk>"):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}

        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        sent = " ".join(token_string(i) for i in tensor if i != self.eos())
        if bpe_symbol is not None:
            sent = (sent + " ").replace(bpe_symbol, "").rstrip()
        return sent

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        threshold_nwords = len(new_symbols)
        if padding_factor > 1:
            i = 0
            while threshold_nwords % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(0)
                i += 1
                threshold_nwords += 1

        assert len(new_symbols) % padding_factor == 0
        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, "r", encoding="utf-8") as fd:
                        return cls.load(fd)
                else:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except Exception:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )

        d = cls()
        for line in f.readlines():
            idx = line.rfind(" ")
            word = line[:idx]
            count = int(line[idx + 1 :])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d

    def save(self, f):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for symbol, count in zip(
            self.symbols[self.nspecial :], self.count[self.nspecial :]
        ):
            print("{} {}".format(symbol, count), file=f)

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t


class SmilesVocabulary(Vocabulary):
    def __init__(self, pad="<pad>", eos="</s>", unk="<unk>", go="<go>"):
        self.unk_word, self.pad_word, self.eos_word, self.go_word = (
            unk,
            pad,
            eos,
            go,
        )
        self.symbols = []
        self.count = []
        self.indices = {}

        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.go_index = self.add_symbol(go)
        self.nspecial = len(self.symbols)
        for token in self.__get_smile_tokens():
            self.add_symbol(token)

    def __get_smile_tokens(self):
        SMILE_TOKENS = [
            "S",
            "O",
            "2",
            "n",
            "l",
            "F",
            "H",
            "C",
            "o",
            "5",
            "r",
            "s",
            "=",
            "6",
            "[",
            "N",
            "4",
            "c",
            "-",
            "3",
            ")",
            "#",
            "]",
            "B",
            "(",
            "1",
        ]
        return SMILE_TOKENS

    def finalize(self, threshold=-1, nwords=-1, padding_factor=1):
        super(SmilesVocabulary, self).finalize(
            threshold=threshold, nwords=nwords, padding_factor=padding_factor
        )

    def go(self):
        """GO index."""
        return self.go_index

    @classmethod
    def load(cls, f=None, ignore_utf_errors=False):
        """Load function for SMILE data.

        Ignore the file and just initialize the vocab.
        """
        return cls()
