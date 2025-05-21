import unittest
import os
import shutil
import json
import torch

from pref_opt_for_mols.dataset.vocabulary import SmilesVocabulary
from pref_opt_for_mols.dataset.tokenizer import SmilesTokenizer

class TestSmilesTokenizer(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_tokenizer_temp_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        # Use a minimal, predictable vocabulary for testing
        self.custom_symbols = ['<pad>', '<unk>', '<go>', '<eos>', 'C', 'N', 'O', '(', ')', '=', '#', '1', '2', '[', ']', 'H', 'c', 'n', 'o', 's']
        self.smiles_vocab = SmilesVocabulary(symbols=self.custom_symbols)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_init_with_vocabulary(self):
        tokenizer = SmilesTokenizer(vocabulary=self.smiles_vocab)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer.vocabulary, self.smiles_vocab)
        self.assertIsNone(tokenizer.hf_tokenizer) # No vocab_file provided

    def test_02_save_and_load_pretrained_custom_vocab(self):
        # 1. Initialize with custom SmilesVocabulary and save it
        tokenizer_to_save = SmilesTokenizer(vocabulary=self.smiles_vocab)
        
        save_path = os.path.join(self.test_dir, "custom_tokenizer_save")
        tokenizer_to_save.save_pretrained(save_path)

        self.assertTrue(os.path.exists(os.path.join(save_path, "vocab.json")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "merges.txt"))) # even if empty
        self.assertTrue(os.path.exists(os.path.join(save_path, "tokenizer_config.json")))
        
        # Check if internal hf_tokenizer got initialized after saving
        self.assertIsNotNone(tokenizer_to_save.hf_tokenizer, "Internal HF tokenizer should be initialized after saving.")

        # 2. Load the saved tokenizer using from_pretrained class method
        loaded_tokenizer = SmilesTokenizer.from_pretrained(save_path, vocabulary_obj=self.smiles_vocab) # Pass vocab obj for full consistency
        self.assertIsNotNone(loaded_tokenizer)
        self.assertIsNotNone(loaded_tokenizer.hf_tokenizer)
        self.assertEqual(len(loaded_tokenizer.hf_tokenizer.get_vocab()), len(self.custom_symbols))
        
        # Check special tokens mapping in loaded HF tokenizer
        self.assertEqual(loaded_tokenizer.hf_tokenizer.bos_token, self.smiles_vocab.go_word)
        self.assertEqual(loaded_tokenizer.hf_tokenizer.eos_token, self.smiles_vocab.eos_word)
        self.assertEqual(loaded_tokenizer.hf_tokenizer.pad_token, self.smiles_vocab.pad_word)
        self.assertEqual(loaded_tokenizer.hf_tokenizer.unk_token, self.smiles_vocab.unk_word)

        # 3. Load the saved tokenizer by passing vocab_file to __init__
        vocab_file = os.path.join(save_path, "vocab.json")
        merges_file = os.path.join(save_path, "merges.txt")
        loaded_tokenizer_init = SmilesTokenizer(vocabulary=self.smiles_vocab, vocab_file=vocab_file, merges_file=merges_file)
        self.assertIsNotNone(loaded_tokenizer_init)
        self.assertIsNotNone(loaded_tokenizer_init.hf_tokenizer)
        self.assertEqual(len(loaded_tokenizer_init.hf_tokenizer.get_vocab()), len(self.custom_symbols))
        self.assertEqual(loaded_tokenizer_init.hf_tokenizer.bos_token, self.smiles_vocab.go_word)


    def test_03_tokenization_and_encoding_hf_mode(self):
        # Save and load to ensure hf_tokenizer is active
        tokenizer = SmilesTokenizer(vocabulary=self.smiles_vocab)
        save_path = os.path.join(self.test_dir, "hf_mode_tokenizer")
        tokenizer.save_pretrained(save_path) # This initializes tokenizer.hf_tokenizer
        
        # Re-initialize by loading to ensure it's in HF mode for this test instance
        # This is better than relying on the state of 'tokenizer' after save_pretrained
        loaded_tokenizer = SmilesTokenizer.from_pretrained(save_path, vocabulary_obj=self.smiles_vocab)

        smi1 = "CCO"
        smi2 = "N=C=O"

        # Test tokenize
        tokens_smi1_enclosed = loaded_tokenizer.tokenize(smi1, enclose=True)
        self.assertEqual(tokens_smi1_enclosed, [[self.smiles_vocab.go_word, 'C', 'C', 'O', self.smiles_vocab.eos_word]])
        
        tokens_smi1_not_enclosed = loaded_tokenizer.tokenize(smi1, enclose=False)
        self.assertEqual(tokens_smi1_not_enclosed, [['C', 'C', 'O']])

        # Test encode
        encoded_smi1_enclosed = loaded_tokenizer.encode(smi1, enclose=True, aslist=True)
        expected_ids_smi1_enclosed = [
            self.smiles_vocab.index(self.smiles_vocab.go_word),
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index('O'),
            self.smiles_vocab.index(self.smiles_vocab.eos_word)
        ]
        self.assertEqual(encoded_smi1_enclosed, [expected_ids_smi1_enclosed])

        encoded_smi1_not_enclosed_torch = loaded_tokenizer.encode(smi1, enclose=False, aslist=False)
        expected_ids_smi1_not_enclosed = [
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index('O')
        ]
        self.assertTrue(torch.equal(encoded_smi1_not_enclosed_torch[0], torch.tensor(expected_ids_smi1_not_enclosed, dtype=torch.long)))

        # Test batch encoding
        smiles_batch = [smi1, smi2]
        encoded_batch_aslist = loaded_tokenizer.encode(smiles_batch, enclose=True, aslist=True)
        expected_ids_smi2_enclosed = [
            self.smiles_vocab.index(self.smiles_vocab.go_word),
            self.smiles_vocab.index('N'),
            self.smiles_vocab.index('='),
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index('='),
            self.smiles_vocab.index('O'),
            self.smiles_vocab.index(self.smiles_vocab.eos_word)
        ]
        self.assertEqual(encoded_batch_aslist, [expected_ids_smi1_enclosed, expected_ids_smi2_enclosed])

    def test_04_decoding_and_detokenization_hf_mode(self):
        tokenizer = SmilesTokenizer(vocabulary=self.smiles_vocab)
        save_path = os.path.join(self.test_dir, "hf_decode_tokenizer")
        tokenizer.save_pretrained(save_path)
        loaded_tokenizer = SmilesTokenizer.from_pretrained(save_path, vocabulary_obj=self.smiles_vocab)

        smi1 = "CCO"
        smi2 = "N=C=O"
        
        # Encode first (using lists of IDs for decode)
        ids_smi1_enclosed_list = [
            self.smiles_vocab.index(self.smiles_vocab.go_word),
            self.smiles_vocab.index('C'), self.smiles_vocab.index('C'), self.smiles_vocab.index('O'),
            self.smiles_vocab.index(self.smiles_vocab.eos_word)
        ]
        ids_smi2_enclosed_list = [
            self.smiles_vocab.index(self.smiles_vocab.go_word),
            self.smiles_vocab.index('N'), self.smiles_vocab.index('='), self.smiles_vocab.index('C'),
            self.smiles_vocab.index('='), self.smiles_vocab.index('O'),
            self.smiles_vocab.index(self.smiles_vocab.eos_word)
        ]
        
        # Test decode
        decoded_smiles = loaded_tokenizer.decode([ids_smi1_enclosed_list, ids_smi2_enclosed_list], skip_special_tokens=True)
        # HF decode might produce slightly different strings for token joining, e.g. no spaces for GPT2Tokenizer
        # The key is that the chemical representation is correct.
        self.assertEqual(decoded_smiles, [smi1, smi2]) # Assumes simple concatenation for these tokens

        decoded_smiles_with_special = loaded_tokenizer.decode([ids_smi1_enclosed_list], skip_special_tokens=False)
        expected_smi1_with_special = self.smiles_vocab.go_word + smi1 + self.smiles_vocab.eos_word
        self.assertEqual(decoded_smiles_with_special, [expected_smi1_with_special])
        
        # Test detokenize (from list of token strings)
        tokens_smi1_enclosed = [self.smiles_vocab.go_word, 'C', 'C', 'O', self.smiles_vocab.eos_word]
        detokenized_smi1 = loaded_tokenizer.detokenize([tokens_smi1_enclosed], include_control_tokens=False, truncate_at_end_token=True)
        self.assertEqual(detokenized_smi1, [smi1]) # After stripping go/eos

        detokenized_smi1_with_special = loaded_tokenizer.detokenize([tokens_smi1_enclosed], include_control_tokens=True)
        self.assertEqual(detokenized_smi1_with_special, [expected_smi1_with_special])

    def test_05_fallback_mode_tokenization_encoding(self):
        # Tokenizer without vocab_file -> fallback mode (no hf_tokenizer)
        tokenizer = SmilesTokenizer(vocabulary=self.smiles_vocab)
        self.assertIsNone(tokenizer.hf_tokenizer)

        smi1 = "CCO"
        # Test tokenize (fallback) - uses regex
        tokens_smi1_enclosed = tokenizer.tokenize(smi1, enclose=True)
        # Regex based tokenizer might produce different tokenization for some cases, but for simple SMILES:
        expected_tokens_fallback = [[self.smiles_vocab.go_word] + list(smi1) + [self.smiles_vocab.eos_word]]
        # The custom_symbols based regex will tokenize C, C, O correctly.
        self.assertEqual(tokens_smi1_enclosed, [[self.smiles_vocab.go_word, 'C', 'C', 'O', self.smiles_vocab.eos_word]])


        # Test encode (fallback)
        encoded_smi1_enclosed = tokenizer.encode(smi1, enclose=True, aslist=True)
        expected_ids_smi1_enclosed = [
            self.smiles_vocab.index(self.smiles_vocab.go_word),
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index('O'),
            self.smiles_vocab.index(self.smiles_vocab.eos_word)
        ]
        self.assertEqual(encoded_smi1_enclosed, [expected_ids_smi1_enclosed])

    def test_06_fallback_mode_decoding_detokenization(self):
        tokenizer = SmilesTokenizer(vocabulary=self.smiles_vocab)
        self.assertIsNone(tokenizer.hf_tokenizer)

        smi1 = "CCO"
        ids_smi1_enclosed_list = [
            self.smiles_vocab.index(self.smiles_vocab.go_word),
            self.smiles_vocab.index('C'), self.smiles_vocab.index('C'), self.smiles_vocab.index('O'),
            self.smiles_vocab.index(self.smiles_vocab.eos_word)
        ]
        # Test decode (fallback)
        decoded_smiles = tokenizer.decode([ids_smi1_enclosed_list], skip_special_tokens=True)
        self.assertEqual(decoded_smiles, [smi1])

        # Test detokenize (fallback)
        tokens_smi1_enclosed = [self.smiles_vocab.go_word, 'C', 'C', 'O', self.smiles_vocab.eos_word]
        detokenized_smi1 = tokenizer.detokenize([tokens_smi1_enclosed], include_control_tokens=False, truncate_at_end_token=True)
        self.assertEqual(detokenized_smi1, [smi1])

    def test_07_tokens_to_smiles(self):
        tokenizer = SmilesTokenizer(vocabulary=self.smiles_vocab)
        save_path = os.path.join(self.test_dir, "tts_tokenizer")
        tokenizer.save_pretrained(save_path) # Init HF tokenizer
        loaded_tokenizer = SmilesTokenizer.from_pretrained(save_path, vocabulary_obj=self.smiles_vocab)

        ids_with_unk = [
            self.smiles_vocab.index(self.smiles_vocab.go_word),
            self.smiles_vocab.index('C'),
            self.smiles_vocab.index(self.smiles_vocab.unk_word), # UNK token
            self.smiles_vocab.index('O'),
            self.smiles_vocab.index(self.smiles_vocab.eos_word)
        ]
        smiles_output = loaded_tokenizer.tokens_to_smiles([ids_with_unk], skip_special_tokens=True)
        # <unk> should be replaced by empty string
        self.assertEqual(smiles_output, ["CO"]) 

        # Test with fallback
        tokenizer_fallback = SmilesTokenizer(vocabulary=self.smiles_vocab)
        smiles_output_fallback = tokenizer_fallback.tokens_to_smiles([ids_with_unk], skip_special_tokens=True)
        self.assertEqual(smiles_output_fallback, ["CO"])


if __name__ == '__main__':
    unittest.main()
