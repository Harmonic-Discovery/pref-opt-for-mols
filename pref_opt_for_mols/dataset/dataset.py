from torch.utils.data import Dataset
from rdkit import Chem
import torch
import random
from .tokenizer import SmilesTokenizer
import random


def _randomize_smiles(smiles):
    """
    randomly permute the order of a smiles string, for data augmentation
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True
    )


class NextTokenSmilesDataset(Dataset):
    """
    Dataset for training autoregressive models - inputs are sequences of tokens, outputs are same sequence shifted by 1 token

    Arguments
        smiles (list of smiles strings)
        scaffolds (list of smiles strings representing scaffolds, optional)
        tokenizer (SmilesTokenizer object)
        enclose (bool): whether or not to add 'go' and 'eos' tokens at the beginning/end of the string
        seq_len (int): length to pad each sequence to
        aug_prob (int): w/ probability aug_prob a smiles string will be randomly permuted (by permuting the atom labelings in rdkit)
    """

    def __init__(
        self,
        smiles,
        scaffolds=None,
        tokenizer: SmilesTokenizer = SmilesTokenizer(),
        enclose=True,
        seq_len=128,
        aug_prob=0.0,
    ):
        self.smiles = smiles
        self.scaffolds = scaffolds
        self.tokenizer = tokenizer
        self.enclose = enclose
        self.seq_len = seq_len
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]

        if random.random() < self.aug_prob:
            smiles = _randomize_smiles(smiles)

        tokens = self.tokenizer.encode([smiles], enclose=self.enclose, aslist=True)[0]
        length = len(tokens) - 1

        if self.seq_len is not None:
            tokens = tokens[: self.seq_len]
            padding = [
                self.tokenizer.vocabulary.pad_index
                for _ in range(self.seq_len - len(tokens))
            ]
            tokens.extend(padding)

        input = tokens[:-1]
        output = tokens[1:]

        return {
            "input": torch.tensor(input, dtype=torch.long),
            "output": torch.tensor(output, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
