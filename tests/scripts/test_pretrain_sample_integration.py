import unittest
import os
import shutil
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Imports from the project
from pref_opt_for_mols.models.gpt import GPT, GPTLightning
from pref_opt_for_mols.dataset.tokenizer import SmilesTokenizer, SmilesVocabulary
from pref_opt_for_mols.dataset.smiles_dataset import NextTokenSmilesDataset
import pytorch_lightning as pl

# Minimal config for integration test
INTEGRATION_TEST_CONFIG = {
    "model_path": "test_integration_temp_dir/model_output",
    "reference_path": "test_integration_temp_dir/model_output", # For GPT.load_from_checkpoint consistency
    "seq_len": 32, # Shorter sequence length for faster test
    "model_type": "gpt-nano", # Smallest model
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "lr_decay": False, # No complex LR scheduling for a tiny test
    "learning_rate": 1e-4,
    "batch_size": 2, # Tiny batch size
    "max_epochs": 1, # Minimal epochs
    "grad_norm_clip": 1.0,
    # vocab_size and block_size will be set dynamically
}

TINY_SMILES_DATA = ["CCO", "NCC", "OCC(N)=O", "C1=CC=CN=C1", "CC(=O)O"] # 5 short SMILES

class TestPretrainSampleIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_integration_temp_dir"
        self.model_output_path = os.path.join(self.test_dir, "model_output")
        os.makedirs(self.model_output_path, exist_ok=True)

        self.config = INTEGRATION_TEST_CONFIG.copy()
        self.config["model_path"] = self.model_output_path
        self.config["reference_path"] = self.model_output_path # DPO/finetune scripts might use this

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_pretrain_and_sample_integration(self):
        # 1. Pretraining Phase (simplified from scripts/pretrain.py)
        
        #   a. Initialize Tokenizer and Vocabulary
        smiles_vocab = SmilesVocabulary(symbols=None) # Use default symbols, will be built from TINY_SMILES_DATA effectively by dataset if not prepopulated
                                                      # For more control, can pre-populate:
                                                      # all_chars = set(''.join(TINY_SMILES_DATA)) | {'<pad>', '<unk>', '<go>', '<eos>'}
                                                      # smiles_vocab = SmilesVocabulary(symbols=list(all_chars))
        # For this test, let's make it more robust by defining a small vocab that covers the test data
        test_symbols = ['<pad>', '<unk>', '<go>', '<eos>', 'C', 'N', 'O', '(', ')', '=', '1']
        smiles_vocab = SmilesVocabulary(symbols=test_symbols)
        tokenizer = SmilesTokenizer(vocabulary=smiles_vocab)

        self.config["vocab_size"] = len(smiles_vocab.symbols)
        self.config["block_size"] = self.config["seq_len"] - 1
        
        #   b. Save tokenizer (as pretrain.py does)
        tokenizer.save_pretrained(self.model_output_path)
        self.assertTrue(os.path.exists(os.path.join(self.model_output_path, "vocab.json")))

        #   c. Create Dataset and DataLoader
        #      NextTokenSmilesDataset will use the tokenizer passed to it.
        #      It also builds its own vocabulary if tokenizer is None, but here we provide one.
        train_dataset = NextTokenSmilesDataset(
            smiles_list=TINY_SMILES_DATA,
            tokenizer=tokenizer, # Pass the initialized tokenizer
            enclose=True,
            seq_len=self.config["seq_len"],
            aug_prob=0.0,
        )
        # Check if dataset's internal tokenizer matches
        self.assertEqual(train_dataset.tokenizer.vocabulary.symbols, smiles_vocab.symbols)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True
        )

        #   d. Initialize GPTLightning model
        model_lightning = GPTLightning(config=self.config.copy()) # Pass a copy in case PL modifies it

        #   e. Run a minimal training loop
        #      Using PyTorch Lightning Trainer for a few steps
        #      Limit steps/epochs for speed. `limit_train_batches=1` ensures only 1 batch per epoch.
        trainer = pl.Trainer(
            default_root_dir=self.model_output_path,
            max_epochs=self.config["max_epochs"],
            limit_train_batches=2, # Train on 2 batches only
            limit_val_batches=0, # No validation for this quick test
            enable_checkpointing=True, # Need checkpoint for sampling
            callbacks=[pl.callbacks.ModelCheckpoint(dirpath=self.model_output_path, filename="model")], # Save as model.ckpt
            logger=False, # Disable logging for tests
            enable_progress_bar=False,
            accelerator="cpu" # Run on CPU for test portability
        )
        
        trainer.fit(model_lightning, train_loader)
        
        # Check if checkpoint is saved
        # ModelCheckpoint saves with epoch and step, e.g. "model-epoch=0-step=2.ckpt"
        # For simplicity, let's find a .ckpt file. The callback also makes a "model.ckpt" symlink/copy.
        self.assertTrue(os.path.exists(os.path.join(self.model_output_path, "model.ckpt")),
                        f"Checkpoint 'model.ckpt' not found in {os.listdir(self.model_output_path)}")

        # Save the config file (as pretrain.py does)
        with open(os.path.join(self.model_output_path, "config.json"), "w") as f:
            json.dump(self.config, f)


        # 2. Sampling Phase (simplified from scripts/sample.py)

        #   a. Load configuration (already have self.config, but sample.py loads it)
        with open(os.path.join(self.model_output_path, "config.json")) as f:
            loaded_run_config = json.load(f)
        
        self.assertEqual(loaded_run_config["vocab_size"], self.config["vocab_size"])

        #   b. Load the trained model using GPT.load_from_checkpoint
        #      This method now also loads the tokenizer.
        checkpoint_file_path = os.path.join(self.model_output_path, "model.ckpt")
        
        # Ensure the model is loaded to CPU for testing
        # The GPT.load_from_checkpoint method should handle the device placement for the model,
        # and also load the tokenizer.
        loaded_gpt_model = GPT.load_from_checkpoint(
            config=loaded_run_config, # Use the config saved by pretraining part
            ckpt_path=checkpoint_file_path,
            device="cpu", # Explicitly load to CPU
            disable_dropout=True
        )
        loaded_gpt_model.to("cpu") # Ensure it's on CPU

        self.assertIsNotNone(loaded_gpt_model, "Failed to load GPT model from checkpoint.")
        self.assertIsNotNone(loaded_gpt_model.tokenizer, "Tokenizer was not loaded with the GPT model.")
        self.assertIsNotNone(loaded_gpt_model.tokenizer.hf_tokenizer, "HF Tokenizer instance not found in loaded model's tokenizer.")
        
        # Check if loaded tokenizer vocab matches original
        self.assertEqual(len(loaded_gpt_model.tokenizer.hf_tokenizer.get_vocab()), len(smiles_vocab.symbols))


        #   c. Generate samples
        num_samples_to_generate = 3
        max_new_tokens_sampling = 10 # Short samples
        
        # The sample method is on the GPT instance (not GPTLightning)
        sampled_smiles_list = loaded_gpt_model.sample(
            n_batch=num_samples_to_generate,
            max_new_tokens=max_new_tokens_sampling
        )

        #   d. Basic checks on samples
        self.assertIsInstance(sampled_smiles_list, list)
        self.assertEqual(len(sampled_smiles_list), num_samples_to_generate)
        print(f"Integration test generated SMILES: {sampled_smiles_list}")
        for smi in sampled_smiles_list:
            self.assertIsInstance(smi, str)
            # Further checks could involve trying to parse with RDKit,
            # but for a basic integration test, non-empty string is a start.
            # Given the tiny training, output might be garbage, but process should work.
            # self.assertTrue(len(smi) > 0) # Could be empty if all are special tokens & skipped

if __name__ == '__main__':
    unittest.main()
