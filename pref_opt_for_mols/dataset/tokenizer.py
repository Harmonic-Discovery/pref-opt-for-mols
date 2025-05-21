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
from transformers import GPT2Tokenizer
import os # For path manipulation in save_pretrained

from .vocabulary import Vocabulary, SmilesVocabulary

Tokens = List[str]


class SmilesTokenizer:
    """
    Smiles Tokenizer
    """

    def __init__(
        self,
        vocabulary: SmilesVocabulary = None, # Expecting SmilesVocabulary specifically
        vocab_file: Optional[str] = None, # Path to vocab.json
        merges_file: Optional[str] = None, # Path to merges.txt
    ) -> None:
        if vocabulary is None and vocab_file is None:
            # Default behavior: use a standard SmilesVocabulary
            warnings.warn("Initializing SmilesTokenizer with default SmilesVocabulary. For Hugging Face compatibility, provide vocab_file and merges_file or save a vocabulary first.")
            self.vocabulary = SmilesVocabulary() # Keep instance for symbol reference if needed
            # This path won't work unless a default HF tokenizer is set up, 
            # or if self.vocabulary is used to create vocab files on the fly (complex).
            # For now, let's assume if no files, it's for non-HF use or pre-HF setup.
            self.hf_tokenizer = None # Will be set if vocab_file is provided or after saving
        elif vocab_file:
            # Load Hugging Face tokenizer from provided files
            try:
                self.hf_tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merges_file if merges_file else None)
                # Special tokens need to be explicitly set if not in vocab file or if defaults are desired
                # We'll use the SmileVocabulary definitions for consistency
                _smiles_vocab_for_special_tokens = vocabulary if vocabulary else SmilesVocabulary()
                self.hf_tokenizer.add_special_tokens({
                    'bos_token': _smiles_vocab_for_special_tokens.go_word,
                    'eos_token': _smiles_vocab_for_special_tokens.eos_word,
                    'pad_token': _smiles_vocab_for_special_tokens.pad_word,
                    'unk_token': _smiles_vocab_for_special_tokens.unk_word,
                })
                # GPT-2 often uses eos_token as pad_token
                if self.hf_tokenizer.pad_token is None and self.hf_tokenizer.eos_token is not None:
                     self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

            except Exception as e:
                warnings.warn(f"Failed to load GPT2Tokenizer from {vocab_file}: {e}. Falling back to basic SmilesVocabulary.")
                self.hf_tokenizer = None
            self.vocabulary = vocabulary if vocabulary else SmilesVocabulary() # Keep for reference
        else: # vocabulary is provided, but no vocab_file (pre-HF conversion setup)
            self.vocabulary = vocabulary
            self.hf_tokenizer = None
        
        self._re: Optional[Pattern] = None # Retain for original tokenization logic if hf_tokenizer is not used

    @property
    def re(self) -> Pattern:
        """Tokens Regex Object.
        Only used if hf_tokenizer is not available.

        :return: Tokens Regex Object
        """
        if not self._re and self.vocabulary: # Ensure vocabulary exists
            self._re = self._get_compiled_regex(self.vocabulary.symbols)
        return self._re

    def tokenize(self, smiles: Union[str, List[str]], enclose: bool = True) -> List[List[str]]:
        """
        Convert list of SMILES strings to list of lists containing tokens for each.
        Uses Hugging Face tokenizer if available.
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        tokenized_data = []
        if self.hf_tokenizer:
            for smi in smiles:
                # HF tokenizer's `tokenize` method typically doesn't add special tokens by default.
                # We need to handle `enclose` manually if `tokenize` is used directly.
                # Alternatively, use `encode` and then `convert_ids_to_tokens`.
                # For simplicity, let's adjust based on `encode` which handles special tokens better.
                
                # The hf_tokenizer.encode() adds special tokens if add_special_tokens=True (default)
                # So, if enclose=True, we want BOS and EOS.
                # If enclose=False, we don't want them.
                # GPT2Tokenizer adds bos_token at the beginning and eos_token at the end by default with encode.
                
                if enclose:
                    # Ensure BOS and EOS are added. This is typical for hf_tokenizer.encode()
                    # We might need to prepend/append manually if hf_tokenizer is configured not to.
                    # Assuming hf_tokenizer correctly adds BOS/EOS if configured.
                    # Let's check the config of hf_tokenizer for bos and eos tokens
                    tokens = self.hf_tokenizer.tokenize(smi, add_special_tokens=False) # Get raw tokens first
                    # Manually add if enclose is true, using the special tokens defined in hf_tokenizer
                    tokens = ([self.hf_tokenizer.bos_token] + tokens + [self.hf_tokenizer.eos_token]) \
                                if self.hf_tokenizer.bos_token and self.hf_tokenizer.eos_token else tokens

                else: # Not enclose
                    tokens = self.hf_tokenizer.tokenize(smi, add_special_tokens=False)
                tokenized_data.append(tokens)
        else: # Fallback to original regex-based tokenization
            warnings.warn("Hugging Face tokenizer not available. Falling back to regex-based tokenization.")
            for smi in smiles:
                if not self.re:
                    raise RuntimeError("Regex pattern not compiled. Ensure vocabulary is loaded.")
                _tokens = self.re.findall(smi)
                if enclose:
                    tokenized_data.append(
                        [self.vocabulary.go_word] + _tokens + [self.vocabulary.eos_word]
                    )
                else:
                    tokenized_data.append(_tokens)
        return tokenized_data

    def encode(self, smiles: Union[str, List[str]], enclose: bool = True, aslist: bool = False, add_special_tokens: Optional[bool] = None):
        """
        Convert a list of SMILES strings to list of tensors (or lists) containing token indices.
        Uses Hugging Face tokenizer if available.
        `add_special_tokens` overrides `enclose` for HF tokenizer if provided.
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        ids_list = []
        if self.hf_tokenizer:
            # Determine if special tokens should be added.
            # If add_special_tokens is explicitly passed, it takes precedence.
            # Otherwise, `enclose` determines it (True means add BOS/EOS).
            _add_special_tokens = add_special_tokens if add_special_tokens is not None else enclose

            for smi in smiles:
                # `encode` method of HF tokenizers typically returns a list of integers (token IDs)
                # It also handles special tokens based on its configuration or `add_special_tokens` argument.
                encoded_ids = self.hf_tokenizer.encode(smi, add_special_tokens=_add_special_tokens, return_tensors=None) # No PyTorch tensors yet
                
                if not aslist:
                    encoded_ids = torch.tensor(encoded_ids, dtype=torch.long)
                ids_list.append(encoded_ids)
        else: # Fallback to original method
            warnings.warn("Hugging Face tokenizer not available. Falling back to original encode method.")
            tokenized_smi_list = self.tokenize(smiles, enclose=enclose) # Uses the old tokenize logic
            for tokens in tokenized_smi_list:
                ids_tensor = []
                for token in tokens:
                    ids_tensor.append(self.vocabulary.index(token)) # Uses SmilesVocabulary
                if not aslist:
                    ids_tensor = torch.tensor(ids_tensor, dtype=torch.long)
                ids_list.append(ids_tensor)
        return ids_list

    def detokenize(
        self,
        token_data: Union[List[List[str]], List[torch.Tensor], List[List[int]]], # Can be tokens or ids
        include_control_tokens: bool = False, # HF decode has skip_special_tokens
        include_end_of_line_token: bool = False, # Custom post-processing
        truncate_at_end_token: bool = False, # HF decode can handle EOS if properly configured
    ) -> List[str]:
        """
        Detokenizes lists of tokens (or decodes token IDs) into SMILES strings.
        Uses Hugging Face tokenizer's decode method if available.
        """
        if not token_data:
            return []

        strings = []
        if self.hf_tokenizer:
            # If input is token IDs (tensors or list of lists of ints)
            # This case should primarily be handled by the `decode` method.
            # `detokenize` historically was for joining token strings.
            if isinstance(token_data[0], torch.Tensor) or \
               (isinstance(token_data[0], list) and token_data[0] and isinstance(token_data[0][0], int)):
                warnings.warn("`detokenize` called with token IDs. Redirecting to `decode`.")
                # `skip_special_tokens` in `decode` is equivalent to `not include_control_tokens`
                # `truncate_at_end_token` is implicitly handled by `skip_special_tokens=True` for EOS.
                strings = self.decode(token_data, skip_special_tokens=not include_control_tokens)

            # If input is list of lists of token strings
            elif isinstance(token_data[0], list) and (not token_data[0] or isinstance(token_data[0][0], str)):
                processed_token_lists = []
                for tokens_list in token_data: # iterate over each list of string tokens in the batch
                    _tokens = tokens_list.copy() # work on a copy
                    if not include_control_tokens: # Manual stripping of special *string* tokens
                        special_token_strings = [st for st in [self.hf_tokenizer.bos_token, self.hf_tokenizer.eos_token, self.hf_tokenizer.pad_token, self.hf_tokenizer.unk_token] if st is not None]
                        _tokens = [t for t in _tokens if t not in special_token_strings]
                    
                    if truncate_at_end_token and self.hf_tokenizer.eos_token in _tokens:
                         eos_idx = _tokens.index(self.hf_tokenizer.eos_token)
                         _tokens = _tokens[:eos_idx] # Truncate before EOS, EOS itself is usually skipped by decode or handled by include_control_tokens

                    processed_token_lists.append(_tokens)
                # `convert_tokens_to_string` typically works on a single list of tokens.
                # It also does not usually handle special tokens by default.
                strings = [self.hf_tokenizer.convert_tokens_to_string(t_list) for t_list in processed_token_lists]
            else:
                warnings.warn(f"Unsupported token_data format for hf_tokenizer.detokenize: {type(token_data[0])}. Falling back.")
                # Fallback to original _strip_list logic for safety, ensuring items are strings
                character_lists = [list(map(str,tokens_item)) for tokens_item in token_data]
                character_lists = [
                    self._strip_list(
                        tkns,
                        strip_control_tokens=not include_control_tokens,
                        truncate_at_end_token=truncate_at_end_token,
                    ) for tkns in character_lists
                ]
                strings = ["".join(s) for s in character_lists]
        else: # Fallback to original regex-based tokenization
            warnings.warn("Hugging Face tokenizer not available. Falling back to original detokenize method.")
            character_lists = [list(map(str,tokens_item)) for tokens_item in token_data]
            character_lists = [
                self._strip_list(
                    tkns,
                    strip_control_tokens=not include_control_tokens,
                    truncate_at_end_token=truncate_at_end_token,
                ) for tkns in character_lists
            ]
            strings = ["".join(s) for s in character_lists]

        if include_end_of_line_token:
            strings = [s + "\n" for s in strings]

        return strings

    def decode(self, ids_list: Union[List[torch.Tensor], List[List[int]]], skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = True):
        """
        Decodes lists of encodings (IDs as tensors or lists of ints) back into SMILES strings.
        Uses Hugging Face tokenizer if available.
        `skip_special_tokens` controls whether special tokens (BOS, EOS, PAD) are removed from the output.
        """
        if not ids_list:
            return []

        smiles_list = []
        if self.hf_tokenizer:
            processed_ids_list = []
            for ids_item in ids_list:
                if isinstance(ids_item, torch.Tensor):
                    processed_ids_list.append(ids_item.tolist())
                elif isinstance(ids_item, list):
                    processed_ids_list.append(ids_item)
                else:
                    warnings.warn(f"Unsupported type in ids_list: {type(ids_item)}. Skipping item.")
            
            smiles_list = self.hf_tokenizer.batch_decode(
                processed_ids_list, 
                skip_special_tokens=skip_special_tokens, 
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
        
        else: # Fallback to original method
            warnings.warn("Hugging Face tokenizer not available. Falling back to original decode method.")
            tokenized_smiles_from_ids = []
            for ids_tensor_or_list in ids_list:
                _ids = ids_tensor_or_list.tolist() if isinstance(ids_tensor_or_list, torch.Tensor) else ids_tensor_or_list
                # Ensure vocabulary is accessible and is the custom SmilesVocabulary for index-to-token mapping
                if not self.vocabulary:
                     raise RuntimeError("SmilesVocabulary not available for fallback decode operation.")
                tokens = [self.vocabulary[i] for i in _ids] 
                tokenized_smiles_from_ids.append(tokens)
            
            # Original decode called detokenize with truncate_at_end_token=True.
            # skip_special_tokens=True in HF is similar.
            # include_control_tokens = not skip_special_tokens
            smiles_list = self.detokenize(
                tokenized_smiles_from_ids, 
                include_control_tokens=not skip_special_tokens, 
                truncate_at_end_token=skip_special_tokens 
            ) # This will call the updated detokenize, which might re-warn if hf_tokenizer is still None.
            
        return smiles_list

    def tokens_to_smiles(self, tokens_or_ids_list: Union[List[List[str]], List[torch.Tensor], List[List[int]]], skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = True):
        """
        Convert generated tokens or token IDs to SMILES strings.
        This is a convenience wrapper around the `decode` method.

        Arguments:
            tokens_or_ids_list: List of token lists (strings), or list of token ID tensors/lists.
            skip_special_tokens: Passed to the decode method.
            clean_up_tokenization_spaces: Passed to the decode method.

        Returns:
            list of smiles strings
        """
        # If the input is token strings, it's better to convert them to IDs first if using HF,
        # but decode now handles this. If it's IDs, decode handles it directly.
        # The main job here is to call decode and then potentially do minor cleanup like <unk>.

        smiles_strings = self.decode(
            ids_list=tokens_or_ids_list, # decode now accepts Union type
            skip_special_tokens=skip_special_tokens, 
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        
        # Replace unk_token string representation if it appears in the output.
        # The actual <unk> string might vary based on tokenizer settings.
        # Using self.vocabulary.unk_word or self.hf_tokenizer.unk_token if available.
        unk_replacement_string = "" # Replace <unk> with empty string
        
        unk_token_str = None
        if self.hf_tokenizer and hasattr(self.hf_tokenizer, 'unk_token') and self.hf_tokenizer.unk_token:
            unk_token_str = self.hf_tokenizer.unk_token
        elif self.vocabulary and hasattr(self.vocabulary, 'unk_word') and self.vocabulary.unk_word:
            unk_token_str = self.vocabulary.unk_word
        
        if unk_token_str and unk_token_str != unk_replacement_string : # Avoid replacing "" with ""
             smiles_strings = [smi.replace(unk_token_str, unk_replacement_string) for smi in smiles_strings]
            
        return smiles_strings

    def _strip_list( # This method might be less relevant if HF tokenizer handles stripping, but keep for now.
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

    def save_pretrained(self, save_directory: str):
        """
        Saves the tokenizer (vocabulary) to a directory so it can be reloaded using
        `GPT2Tokenizer.from_pretrained(save_directory)`.
        """
        if self.hf_tokenizer:
            self.hf_tokenizer.save_pretrained(save_directory)
            print(f"Hugging Face tokenizer saved to {save_directory}")
        elif self.vocabulary:
            # Fallback: Save the custom SmilesVocabulary if no HF tokenizer is present.
            # This requires SmilesVocabulary to have a save method, or implement one here.
            # For Option A, the goal is to create vocab.json and merges.txt.
            
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # 1. Create vocab.json
            #    The format is a dictionary of token to id.
            #    GPT2Tokenizer also expects special tokens to be part of this if not added via add_special_tokens.
            vocab_json_path = os.path.join(save_directory, "vocab.json")
            
            # Get all tokens from SmilesVocabulary, including special ones
            # Ensure special tokens are handled correctly, GPT2Tokenizer might add them if they are in its config
            # but not in vocab.json, or vice-versa. It's safer to have them in vocab.json.
            token_to_id = {token: i for i, token in enumerate(self.vocabulary.symbols)}
            
            # Ensure all defined special tokens in SmilesVocabulary are in the vocab map
            # and use their string representations as keys.
            special_tokens_map = {
                "bos_token": self.vocabulary.go_word,
                "eos_token": self.vocabulary.eos_word,
                "unk_token": self.vocabulary.unk_word,
                "pad_token": self.vocabulary.pad_word,
                # Add other special tokens if SmilesVocabulary defines them (e.g., sep_token, cls_token, mask_token)
            }
            
            # Add/update special tokens in the vocab map. Their IDs should be consistent.
            # The order in self.vocabulary.symbols already determines IDs if special words are part of it.
            # If they are not, we need to add them.
            # For simplicity, let's assume SmilesVocabulary.symbols includes all tokens (incl. special).
            # GPT2Tokenizer will typically sort the vocab by ID for its internal use.
            
            current_max_id = len(token_to_id)
            for token_name, token_value in special_tokens_map.items():
                if token_value not in token_to_id:
                    # This case should ideally be handled by SmilesVocabulary ensuring all its tokens are in .symbols
                    # If we add them here, their IDs might not match what SmilesVocabulary.index() would give.
                    # For now, assume `self.vocabulary.symbols` is the source of truth for tokens and their order.
                    # If `SmilesVocabulary.index(token_value)` works, it means they are already in `symbols`.
                    pass # Assuming they are already in symbols from SmilesVocabulary

            with open(vocab_json_path, "w", encoding="utf-8") as f:
                json.dump(token_to_id, f, ensure_ascii=False, indent=2)

            # 2. Create merges.txt
            #    For a simple character-level tokenizer or if not using BPE, this can be minimal or empty.
            #    GPT2 requires it. For SMILES, we might not have complex merges initially.
            #    A common practice for non-BPE or character tokenizers when needing to satisfy GPT2's format
            #    is to list all single characters as "merges" or leave it empty if the tokenizer handles it.
            #    If merges.txt is empty, GPT2Tokenizer might default to character-level tokenization based on vocab.json.
            #    Let's create an empty one for now, or one with basic character pairs if meaningful for SMILES.
            #    For now, an empty merges.txt is safer if we don't have BPE logic.
            #    However, GPT2Tokenizer might expect *some* merges.
            #    A minimal merges.txt could list merges for characters that should stick together, if any.
            #    If we want pure char-level based on vocab.json, an empty merges.txt might lead to issues
            #    or fallback to byte-level BPE.
            #    Let's create a dummy merges.txt that allows individual tokens from vocab.
            #    A simple merges file for a character-based tokenizer would list all pairs of characters
            #    that form the tokens, but since our tokens are sometimes multi-char (e.g., "Cl", "Br"),
            #    a true BPE setup is more complex.
            #    For now, to make GPT2Tokenizer load, we can provide an empty merges file,
            #    or a very basic one. Let's try with an empty one first.
            merges_txt_path = os.path.join(save_directory, "merges.txt")
            with open(merges_txt_path, "w", encoding="utf-8") as f:
                # f.write("#version: 0.2\n") # Optional version line
                pass # Empty merges file

            # 3. Optionally, save tokenizer_config.json if we have special token mappings
            #    GPT2Tokenizer.save_pretrained normally handles this.
            #    We are creating the files manually here for the *first* save if no hf_tokenizer exists.
            tokenizer_config = {
                "model_max_length": 1024, # Default, can be adjusted
                "bos_token": self.vocabulary.go_word,
                "eos_token": self.vocabulary.eos_word,
                "unk_token": self.vocabulary.unk_word,
                "pad_token": self.vocabulary.pad_word,
            }
            tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

            print(f"Custom SmilesVocabulary saved as Hugging Face compatible files in {save_directory}")
            print(f"You can now load this with: tokenizer = GPT2Tokenizer.from_pretrained('{save_directory}') or within SmilesTokenizer by providing vocab_file and merges_file path.")
            # After saving, we could initialize self.hf_tokenizer here
            try:
                self.hf_tokenizer = GPT2Tokenizer(vocab_file=vocab_json_path, merges_file=merges_txt_path)
                # Re-apply special tokens to the newly loaded HF tokenizer
                self.hf_tokenizer.add_special_tokens({
                    'bos_token': self.vocabulary.go_word,
                    'eos_token': self.vocabulary.eos_word,
                    'pad_token': self.vocabulary.pad_word,
                    'unk_token': self.vocabulary.unk_word,
                })
                if self.hf_tokenizer.pad_token is None and self.hf_tokenizer.eos_token is not None:
                     self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
                print("Internal Hugging Face tokenizer initialized from saved files.")
            except Exception as e:
                warnings.warn(f"Failed to initialize internal Hugging Face tokenizer after saving: {e}")

        else:
            warnings.warn("No Hugging Face tokenizer or custom vocabulary to save.")

    # It's more idiomatic for from_pretrained to be a classmethod.
    # However, to fit the current structure, we might have it load into an instance,
    # or the user can call GPT2Tokenizer.from_pretrained directly.
    # For now, __init__ handles loading if vocab_file is provided.
    # A true from_pretrained would look like:
    @classmethod
    def from_pretrained(cls, load_directory: str, vocabulary_obj: SmilesVocabulary = None):
        """
        Loads the tokenizer from a directory.
        `vocabulary_obj` can be provided if the original SmilesVocabulary instance is needed for other metadata.
        """
        vocab_file = os.path.join(load_directory, "vocab.json")
        merges_file = os.path.join(load_directory, "merges.txt") # May or may not exist / be needed
        
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"vocab.json not found in {load_directory}")

        # We pass the file paths to the constructor.
        # The constructor will initialize GPT2Tokenizer.
        # If merges_file doesn't exist, GPT2Tokenizer might complain or use defaults.
        # It's better to ensure merges_file exists, even if empty, if GPT2Tokenizer expects it.
        if not os.path.exists(merges_file):
            warnings.warn(f"merges.txt not found in {load_directory}. GPT2Tokenizer might behave unexpectedly or require it. Creating an empty one for loading.")
            with open(merges_file, "w", encoding="utf-8") as f:
                pass # Create empty merges file

        return cls(vocabulary=vocabulary_obj, vocab_file=vocab_file, merges_file=merges_file)

