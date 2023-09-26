"""
Smiles tokenizer module. 

The class SmilesTokenizer (adapted to be compatible with the Vocabulary object from Fairseq) is used to tokenize 
smiles strings given a SmilesVocabulary (or GeneralSmilesVocabulary). Some of this code is adapted from MolecularAI's
PySMILESutils package (https://github.com/MolecularAI/pysmilesutils).
"""
import re
import json
import warnings
from re import Pattern
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Any

import torch

from .vocabulary import Vocabulary, SmilesVocabulary

Tokens = List[str]


class SmilesTokenizer:
    """
    Smiles Tokenizer
    """

    def __init__(
        self,
        vocabulary: Vocabulary = None,
    ) -> None:
        if vocabulary is None:
            self.vocabulary = SmilesVocabulary()
        else:
            self.vocabulary = vocabulary
        self._re: Optional[Pattern] = None

    @property
    def re(self) -> Pattern:
        """Tokens Regex Object.

        :return: Tokens Regex Object
        """
        if not self._re:
            self._re = self._get_compiled_regex(self.vocabulary.symbols)
        return self._re

    def tokenize(self, smiles: List[str], enclose: bool = True) -> List[List[str]]:
        """
        convert list of smiles strings to list of lists containing tokens for each
        """
        if isinstance(smiles, str):
            # Convert string to a list with one string
            smiles = [smiles]

        tokenized_data = []

        for smi in smiles:
            tokens = self.re.findall(smi)
            if enclose:
                tokenized_data.append(
                    [self.vocabulary.go_word] + tokens + [self.vocabulary.eos_word]
                )
            else:
                tokenized_data.append(tokens)

        return tokenized_data

    def encode(self, smiles: List[str], enclose: bool = True, aslist=False):
        """
        convert a list of smiles strings to list of tensors containing token indices
        """
        if isinstance(smiles, str):
            # Convert string to a list with one string
            smiles = [smiles]

        tokenized_smiles = self.tokenize(smiles, enclose=enclose)
        tokens_lengths = list(map(len, tokenized_smiles))
        ids_list = []

        for tokens, length in zip(tokenized_smiles, tokens_lengths):
            ids_tensor = []  # torch.zeros(length, dtype=torch.long)
            for tdx, token in enumerate(tokens):
                ids_tensor.append(self.vocabulary.index(token))
            if not aslist:
                ids_tensor = torch.tensor(ids_tensor, dtype=torch.long)
            ids_list.append(ids_tensor)

        return ids_list

    def detokenize(
        self,
        token_data: List[List[str]],
        include_control_tokens: bool = False,
        include_end_of_line_token: bool = False,
        truncate_at_end_token: bool = False,
    ) -> List[str]:
        """
        Detokenizes lists of tokens into SMILES by concatenating the token strings.
        """

        character_lists = [tokens.copy() for tokens in token_data]

        character_lists = [
            self._strip_list(
                tokens,
                strip_control_tokens=not include_control_tokens,
                truncate_at_end_token=truncate_at_end_token,
            )
            for tokens in character_lists
        ]

        if include_end_of_line_token:
            for s in character_lists:
                s.append("\n")

        strings = ["".join(s) for s in character_lists]

        return strings

    def decode(self, ids_list: List[torch.Tensor]):
        """
        decodes lists of encodings (ids as tensors) back into smiles strings
        """

        tokenized_smiles = []
        for ids in ids_list:
            if not isinstance(ids, list):
                ids = ids.tolist()

            tokens = [self.vocabulary[i] for i in ids]
            tokenized_smiles.append(tokens)
        smiles = self.detokenize(tokenized_smiles, truncate_at_end_token=True)
        return smiles

    def tokens_to_smiles(self, tokens):
        """
        Convert generated tokens to smiles.

        Arguments:
            tokens: list of tokens

        Returns:
            list of smiles strings
        """
        # convert tokens to smiles
        smiles = self.decode(tokens)
        smiles = [smi.replace("<unk>", "") for smi in smiles]
        return smiles

    def _strip_list(
        self,
        tokens: List[str],
        strip_control_tokens: bool = False,
        truncate_at_end_token: bool = False,
    ) -> List[str]:
        """Cleanup tokens list from control tokens.

        :param tokens: List of tokens
        :param strip_control_tokens: Flag to remove control tokens, defaults to False
        :param truncate_at_end_token: If True truncate tokens after end-token
        """
        if truncate_at_end_token and self.vocabulary.eos_word in tokens:
            end_token_idx = tokens.index(self.vocabulary.eos_word)
            tokens = tokens[: end_token_idx + 1]

        strip_characters: List[str] = [self.vocabulary.pad_word]
        if strip_control_tokens:
            strip_characters.extend([self.vocabulary.go_word, self.vocabulary.eos_word])
        while len(tokens) > 0 and tokens[0] in strip_characters:
            tokens.pop(0)

        while len(tokens) > 0 and tokens[-1] in strip_characters:
            tokens.pop()

        return tokens

    def _get_compiled_regex(self, tokens: List[str]) -> Pattern:
        """Create a Regular Expression Object from a list of tokens and regular expression tokens.

        :param tokens: List of tokens
        :return: Regular Expression Object
        """
        regex_string = r"("  # r"("
        for ix, token in enumerate(tokens):
            processed_token = token
            for special_character in "()[]+*":
                processed_token = processed_token.replace(
                    special_character, f"\{special_character}"
                )
            if ix < len(tokens) - 1:
                regex_string += processed_token + r"|"
            else:
                regex_string += processed_token

        regex_string += r")"
        pattern = re.compile(regex_string)
        return pattern
