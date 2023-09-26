import random
import torch
import numpy as np
from .tokenizer import SmilesTokenizer
from .dataset import _randomize_smiles

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


class PreferencePairDataset(torch.utils.data.Dataset):
    """
    Dataset for constructing DPO-ready preference pairs from labeled data.
    Supports both binary and continuous labels.
    """

    def __init__(
        self,
        data,
        smiles_key="smiles",
        label_key="label",
        aug_prob=0.0, 
        n=int(1e6),
        tokenizer=SmilesTokenizer(),
        seq_len=128,
        enclose=True,
        name=None,
        seed=None,
        binary=True
    ):
        """
        Arguments:
            data: pd.DataFrame containing smiles and labels 
                - Ex: [ {"smiles": "C1CC",  "accept": False } ]
            smiles_key: str, column name for SMILES strings (default: "smiles")
            label_key: str, column name for binary labels (default: "label")
            aug_prob: float, probability of applying SMILES augmentation via 
                random atom relabeling with rdkit (default: 0.0)
            n: int, maximum number of preference pairs to generate (default: 10^6)
            tokenizer: SmilesTokenizer object (default: SmilesTokenizer())
            seq_len: int, length to pad each sequence to (default: 128)
            enclose: bool, whether or not to add 'go' and 'eos' tokens at the 
                beginning/end of the string (default: True)
            name: str, name of dataset (default: None)
            seed: int, random seed for pair generation reproducibility (default: None)
            binary: bool, whether or not labels are binary (default: True)
        """
        if seed is not None:
            self.seed = seed
            state = np.random.get_state()
            np.random.seed(seed)

        self._data = data.dropna()
        self.smiles_key = smiles_key
        self.label_key = label_key
        self.aug_prob = aug_prob
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.enclose = enclose
        self.name = name

        if binary:
            # Generate positive-negative pairs
            pos = self._data[self._data[self.label_key]]
            neg = self._data[~self._data[self.label_key]]

            # Sample positive-negative pairs
            pos_smiles = pos[self.smiles_key].sample(n, replace=True)
            neg_smiles = neg[self.smiles_key].sample(n, replace=True)

            self.smiles_pairs = np.stack([pos_smiles, neg_smiles], axis=1)
        
        else:
            # Generate preference pairs (any two SMILES are ordered if continuous)
            smiles_pairs = self._data[self.smiles_key].sample(2 * n, replace=True)
            self.smiles_pairs = smiles_pairs.values.reshape(-1, 2)

        print(f"Generated {n} preference pairs")
        
        if seed is not None:
            np.random.set_state(state)

    def dataloader(self, *args, **kwargs):
        """
        Gets torch DataLoader for dataset, with padding collate function.

        Arguments:
            dataset: PreferencePairDataset
            *args, **kwargs: arguments to torch.utils.data.DataLoader
        Returns:
            DataLoader object
        """
        kwargs["collate_fn"] = self._collate
        return torch.utils.data.DataLoader(self, *args, **kwargs)

    def _collate(self, batch):
        """
        Collates batch given by self[index].

        Arguments:
            batch: list of dict with keys "positive", "negative", 
                "positive_length", "negative_length"
        Returns:
            dict
        """
        pos, neg, pos_lens, neg_lens = [], [], [], []
        for sample in batch:
            pos.append(sample["positive"])
            neg.append(sample["negative"])
            pos_lens.append(sample["positive_length"])
            neg_lens.append(sample["negative_length"])
        
        data = {"positive": torch.stack(pos),
                "negative": torch.stack(neg),
                "positive_length": torch.tensor(neg_lens),
                "negative_length": torch.tensor(pos_lens)}
        return data

    def __len__(self):
        return len(self.smiles_pairs)
    
    def __getitem__(self, idx):
        """
        Gets input-output pair from dataset.

        Arguments:
            idx: int, index of pair to retrieve
        Returns:
            dict of batch data with below format
            - "positive": (seq_len - 1,) sequence tokens
            - "negative": (seq_len - 1,) sequence tokens
            - "positive_length": scalar length of positive sequence
            - "negative_length": scalar length of negative sequence
        """
        item = {}

        for i, smiles in enumerate(self.smiles_pairs[idx]):
            if random.random() < self.aug_prob:
                smiles = _randomize_smiles(smiles) or smiles

            tokens = self.tokenizer.encode([smiles], enclose=self.enclose, aslist=True)[0]
            length = len(tokens) - 1

            if self.seq_len is not None:
                length = min(length, self.seq_len - 1)
                tokens = tokens[:self.seq_len - 1]
                padding = [
                    self.tokenizer.vocabulary.pad_index
                    for _ in range(self.seq_len - len(tokens) - 1)
                ]
                tokens.extend(padding)

            key = "positive" if i == 0 else "negative"
            item[key] = torch.tensor(tokens, dtype=torch.long)
            item[f"{key}_length"] = length

        return item
