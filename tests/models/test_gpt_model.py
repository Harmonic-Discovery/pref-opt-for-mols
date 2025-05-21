import unittest
import os
import shutil
import json
import torch

from pref_opt_for_mols.models.gpt import GPT
from pref_opt_for_mols.dataset.tokenizer import SmilesTokenizer
from pref_opt_for_mols.dataset.vocabulary import SmilesVocabulary # For creating dummy tokenizer

# Minimal GPT config for testing
# Uses gpt-nano settings from GPT_MODEL_FACTORY
TEST_GPT_CONFIG = {
    "vocab_size": 50, # Will be updated by tokenizer
    "block_size": 63, # seq_len - 1
    "model_type": "gpt-nano", # n_layer=3, n_head=3, n_embd=48
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    # Other params like lr, etc., are not strictly needed for model structure tests
}

class TestGPTModel(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_gpt_model_temp_dir"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a dummy tokenizer and save it for loading tests
        self.custom_symbols = ['<pad>', '<unk>', '<go>', '<eos>', 'C', 'N', 'O', '(', ')', '=', '#'] # Min vocab
        self.smiles_vocab = SmilesVocabulary(symbols=self.custom_symbols)
        self.tokenizer_for_test = SmilesTokenizer(vocabulary=self.smiles_vocab)
        
        self.tokenizer_save_path = os.path.join(self.test_dir, "dummy_tokenizer_for_gpt")
        self.tokenizer_for_test.save_pretrained(self.tokenizer_save_path)
        # Update TEST_GPT_CONFIG vocab_size to match this tokenizer
        self.gpt_config = TEST_GPT_CONFIG.copy()
        self.gpt_config["vocab_size"] = len(self.custom_symbols)


    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_init_from_config(self):
        model = GPT(self.gpt_config)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.transformer) # Should be a GPT2LMHeadModel
        self.assertEqual(model.transformer.config.vocab_size, self.gpt_config["vocab_size"])
        self.assertEqual(model.transformer.config.n_positions, self.gpt_config["block_size"])
        self.assertIsNotNone(model.tokenizer) # Default tokenizer should be initialized

    @unittest.skipIf("GITHUB_ACTIONS" in os.environ, "Skipping Hugging Face model download in CI due to potential network issues.")
    def test_02_init_with_hf_pretrained_model_name(self):
        # This test requires internet access and might be slow.
        # We use a very small standard model. Let's try "sshleifer/tiny-gpt2" if available or a small gpt2 variant.
        # For robustness, this might need to be conditional or use a pre-downloaded model.
        # For now, let's assume "gpt2" can be loaded (it's larger than ideal for a unit test).
        # A truly tiny model like "sshleifer/tiny-gpt2" would be better if it's compatible with GPT2LMHeadModel directly.
        # The load_from_checkpoint method is designed for this.
        
        # We need a config that matches roughly, or let load_from_checkpoint handle it.
        # The GPT.load_from_checkpoint is designed to load HF models by their name.
        # The config passed to load_from_checkpoint is mostly for our internal class structure,
        # the actual HF model config is loaded by from_pretrained.
        
        # We need a config to instantiate GPT first.
        # The vocab size in this config might be overridden by the loaded HF model's config.
        config_for_hf_load = self.gpt_config.copy()

        try:
            # "sshleifer/tiny-gpt2" is a very small distilled model.
            # Using it instead of full "gpt2" for speed and size.
            # model = GPT.load_from_checkpoint(config_for_hf_load, "sshleifer/tiny-gpt2", device="cpu")
            # For this test, let's assume the first argument to load_from_checkpoint should be the model name itself
            # if we are loading a HF pre-trained one.
            # The current `load_from_checkpoint` tries `GPT2LMHeadModel.from_pretrained(ckpt_path)`
            # where `ckpt_path` is the second argument.
            
            # So, we pass the HF model name as `ckpt_path`.
            # The `config` argument is used to initialize `cls(config)` first.
            model = GPT.load_from_checkpoint(config=config_for_hf_load, ckpt_path="sshleifer/tiny-gpt2", device="cpu")
            self.assertIsNotNone(model)
            self.assertIsNotNone(model.transformer)
            # Check if the tokenizer was also loaded (it should be, by HF's from_pretrained)
            # The GPT.load_from_checkpoint now attempts to load OUR SmilesTokenizer from the same path.
            # This will fail for "sshleifer/tiny-gpt2" as there are no SmilesTokenizer files there.
            # It should gracefully warn and keep the default tokenizer or the one from config.
            # For this specific test, we are more interested in the transformer being loaded.
            # The tokenizer part will be tested more thoroughly in test_03.
            self.assertTrue(model.transformer.config.name_or_path == "sshleifer/tiny-gpt2")
        except OSError as e:
            self.skipTest(f"Skipping HF model download test, failed to download: {e}")
        except Exception as e: # Other potential errors during loading
            self.fail(f"Loading HF pretrained model failed with: {e}")


    def test_03_load_from_checkpoint_custom(self):
        # 1. Create a dummy model state_dict (simulating a saved policy)
        #    Initialize a model from config first to get the structure.
        model_to_save = GPT(self.gpt_config)
        dummy_state_dict = model_to_save.state_dict() # Get state_dict of the transformer wrapper

        # 2. Create a dummy checkpoint file (like a PL checkpoint)
        checkpoint_dir = os.path.join(self.test_dir, "dummy_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save tokenizer files into this directory (as pretrain.py would do)
        self.tokenizer_for_test.save_pretrained(checkpoint_dir)
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "vocab.json")))

        dummy_ckpt_path = os.path.join(checkpoint_dir, "model.ckpt")
        # Save the state_dict of the *transformer* as if it's from a PL checkpoint's 'state_dict'
        # The map_state_dict_params in GPT.load_from_checkpoint expects keys like "transformer.xxx"
        # or "gpt.transformer.xxx" if it's the whole GPTLightning model.
        # Let's save the state_dict of `model_to_save.transformer` directly,
        # but map_state_dict_params expects the "transformer." prefix.
        # The `model_to_save.state_dict()` already includes "transformer." prefix.
        torch.save({"state_dict": dummy_state_dict}, dummy_ckpt_path)
        
        # 3. Load using GPT.load_from_checkpoint
        # Pass the *original* config used to save (self.gpt_config matches this)
        loaded_model = GPT.load_from_checkpoint(config=self.gpt_config, ckpt_path=dummy_ckpt_path, device="cpu")
        
        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(loaded_model.transformer)
        # Check if weights seem loaded (e.g., compare one parameter)
        # This is a basic check; a full comparison is too much for a unit test.
        # For example, check the embedding layer's weight.
        original_emb_weight = model_to_save.transformer.get_input_embeddings().weight.data
        loaded_emb_weight = loaded_model.transformer.get_input_embeddings().weight.data
        self.assertTrue(torch.equal(original_emb_weight, loaded_emb_weight))

        # Crucially, test if the tokenizer was loaded
        self.assertIsNotNone(loaded_model.tokenizer)
        self.assertIsNotNone(loaded_model.tokenizer.hf_tokenizer, "HF tokenizer should be loaded from the checkpoint directory.")
        self.assertEqual(len(loaded_model.tokenizer.hf_tokenizer.get_vocab()), len(self.custom_symbols))
        self.assertEqual(loaded_model.tokenizer.hf_tokenizer.bos_token, self.smiles_vocab.go_word)

    def test_04_forward_pass(self):
        model = GPT(self.gpt_config)
        model.to("cpu") # Ensure model is on CPU for test consistency
        
        # Dummy input: batch_size=2, sequence_length=10
        # Token IDs should be within vocab_size
        dummy_input_ids = torch.randint(0, self.gpt_config["vocab_size"], (2, 10), dtype=torch.long)
        
        logits, loss = model.forward(dummy_input_ids, targets=None)
        
        self.assertIsNotNone(logits)
        self.assertIsNone(loss) # No targets provided
        # Expected shape: (batch_size, sequence_length, vocab_size)
        self.assertEqual(logits.shape, (2, 10, self.gpt_config["vocab_size"]))

        # Test with targets
        dummy_targets = torch.randint(0, self.gpt_config["vocab_size"], (2, 10), dtype=torch.long)
        logits_with_loss, loss_val = model.forward(dummy_input_ids, targets=dummy_targets)
        self.assertIsNotNone(logits_with_loss)
        self.assertIsNotNone(loss_val)
        self.assertTrue(torch.is_tensor(loss_val))

    def test_05_sample_method(self):
        # Load a model with a (dummy) trained state and tokenizer
        # For this, we can reuse the setup from test_03_load_from_checkpoint_custom
        checkpoint_dir = os.path.join(self.test_dir, "dummy_checkpoint_for_sample")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.tokenizer_for_test.save_pretrained(checkpoint_dir) # Save tokenizer
        
        # Create a model, save its state dict
        model_to_save_for_sample = GPT(self.gpt_config)
        dummy_state_dict_for_sample = model_to_save_for_sample.state_dict()
        dummy_ckpt_path_for_sample = os.path.join(checkpoint_dir, "model.ckpt")
        torch.save({"state_dict": dummy_state_dict_for_sample}, dummy_ckpt_path_for_sample)

        # Load the model
        model = GPT.load_from_checkpoint(config=self.gpt_config, ckpt_path=dummy_ckpt_path_for_sample, device="cpu")
        model.to("cpu") # Ensure on CPU

        self.assertIsNotNone(model.tokenizer.hf_tokenizer, "Tokenizer must be loaded for sample method to work correctly.")

        n_batch = 2
        max_new_tokens = 5 # Keep it very short
        
        # Ensure the model's tokenizer has the go_index
        self.assertIsNotNone(model.tokenizer.vocabulary.go_index)

        smiles_output = model.sample(n_batch=n_batch, max_new_tokens=max_new_tokens)
        
        self.assertIsInstance(smiles_output, list)
        self.assertEqual(len(smiles_output), n_batch)
        for smi in smiles_output:
            self.assertIsInstance(smi, str)
            # Basic check: starts with GO token (or part of it, if skip_special_tokens=True in decode)
            # and ends with EOS token (or part of it).
            # The sample method in GPT uses tokenizer.decode which by default skips special tokens.
            # So, we expect SMILES content.
            # Given max_new_tokens=5, and GO is 1 token, we expect up to 4 content tokens.
            # A very basic check is that it's not empty if sampling worked.
            # With an untrained model, output might be repetitive or nonsensical.
            print(f"Sampled (test_05): {smi}") # For debugging
            # self.assertTrue(len(smi) > 0) # Might be empty if all are special tokens and skipped.
                                         # Or if it samples PAD early.
            # A more robust check is that it doesn't error out and returns right type.
            # And that the tokens are from its vocabulary.
            self.assertTrue(all(c in self.custom_symbols or c == '' for c in smi.replace(self.smiles_vocab.go_word, '').replace(self.smiles_vocab.eos_word, '')))


    def test_06_logprobs_method(self):
        model = GPT(self.gpt_config)
        model.to("cpu")

        # Dummy input: batch_size=2, sequence_length=10
        dummy_seqs = torch.randint(0, self.gpt_config["vocab_size"], (2, 10), dtype=torch.long)
        dummy_lengths = torch.tensor([10, 8], dtype=torch.long) # Example lengths

        log_probs = model.logprobs(dummy_seqs, dummy_lengths)
        
        self.assertIsNotNone(log_probs)
        self.assertEqual(log_probs.shape, (2,)) # (batch_size,)
        self.assertFalse(torch.isnan(log_probs).any())
        self.assertFalse(torch.isinf(log_probs).any())

if __name__ == '__main__':
    unittest.main()
