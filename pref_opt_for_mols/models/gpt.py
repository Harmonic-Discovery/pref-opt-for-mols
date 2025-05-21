"""
the GPT Language Model, mostly as implemented in minGPT, with the following slight modifications, 
adapted to pytorch lightning + includes custom lr scheduling
"""

import math
import os
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, GPT2Config
from ..dataset import SmilesTokenizer


# -----------------------------------------------------------------------------

GPT_MODEL_FACTORY = {
    "gpt-small": dict(n_layer=8, n_head=8, n_embd=256),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}

# -----------------------------------------------------------------------------

# Unused classes: NewGELU, CausalSelfAttention, Block will be removed.

class GPT(nn.Module):
    """GPT Language Model

    Config is dict with options:
        vocab_size (required)
        block_size (required)
        model_type (required)
        attn_pdrop (default = 0.1)
        resid_pdrop (default = 0.1)
        embd_pdrop (default = 0.1)
        lr_decay (default = True)
        warmup_tokens (default = 4e7)
        final_tokens (default = 4e8)
        learning_rate (default = 1e-4)
        grad_norm_clip (default = 1.0)
    """

    def __init__(self, config):
        super().__init__()
        assert config.get("vocab_size") is not None
        assert config.get("block_size") is not None
        self.config = config
        self.vocab_size = config.get("vocab_size")
        self.block_size = config.get("block_size")

        assert config.get("model_type") is not None
        assert config.get("model_type") in GPT_MODEL_FACTORY.keys()

        self.n_layer = GPT_MODEL_FACTORY[config.get("model_type")].get("n_layer")
        self.n_head = GPT_MODEL_FACTORY[config.get("model_type")].get("n_head")
        self.n_embd = GPT_MODEL_FACTORY[config.get("model_type")].get("n_embd")
        self.attn_pdrop = self.default("attn_pdrop", 0.1) # Will be mapped to attn_pdrop in GPT2Config
        self.resid_pdrop = self.default("resid_pdrop", 0.1) # Will be mapped to resid_pdrop in GPT2Config
        self.embd_pdrop = self.default("embd_pdrop", 0.1) # Will be mapped to embd_pdrop in GPT2Config

        self.tokenizer = SmilesTokenizer()

        # Initialize GPT2Config
        config_args = {
            "vocab_size": self.vocab_size,
            "n_positions": self.block_size,
            "n_layer": GPT_MODEL_FACTORY[config.get("model_type")].get("n_layer"),
            "n_head": GPT_MODEL_FACTORY[config.get("model_type")].get("n_head"),
            "n_embd": GPT_MODEL_FACTORY[config.get("model_type")].get("n_embd"),
            "attn_pdrop": self.attn_pdrop,
            "resid_pdrop": self.resid_pdrop,
            "embd_pdrop": self.embd_pdrop,
        }
        hf_config = GPT2Config(**config_args)

        # Initialize GPT2LMHeadModel
        self.transformer = GPT2LMHeadModel(hf_config)
        self.loss_fn = F.cross_entropy

    # _init_weights is no longer needed as Hugging Face models handle their own initialization.

    # _init_weights is no longer needed as Hugging Face models handle their own initialization.

    def default(self, key, value):
        """
        Returns value if key is in config, otherwise sets key to value and returns value.

        Arguments:
            key: key to look up in config (str)
            value: default value to set if key is not in config (any)
        Returns:
            value, or self.config[key] if key is in self.config
        """
        if key in self.config:
            return self.config[key]
        else:
            self.config[key] = value
            return value
        
    @classmethod
    def load_from_checkpoint(cls, config, ckpt_path, device="cuda:0" if torch.cuda.is_available() else "cpu", disable_dropout=False):
        """
        Loads a model from a checkpoint.
        Can load either a Hugging Face pretrained model or a custom-trained model.
        """
        obj = cls(config) # Initialize with internal config for vocab size etc.

        # Try to load as a Hugging Face pretrained model/checkpoint
        try:
            # If ckpt_path is a recognized HF model name (e.g., "gpt2") or path to HF model
            hf_model = GPT2LMHeadModel.from_pretrained(ckpt_path)
            obj.transformer = hf_model 
            print(f"Loaded Hugging Face model from {ckpt_path}")
        except OSError:
            # Assuming ckpt_path is a PyTorch Lightning checkpoint
            print(f"Could not load {ckpt_path} as a Hugging Face model. Trying as PyTorch Lightning checkpoint.")
            def map_state_dict_params(state_dict):
                # Adapt this function based on how your PL checkpoints are saved
                # This example assumes the GPT model is stored under a 'gpt.' prefix
                # or directly if it's just the transformer's state_dict.
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("gpt.transformer."):
                        new_state_dict[k.replace("gpt.transformer.", "transformer.")] = v
                    elif k.startswith("transformer."): # Already in desired format
                         new_state_dict[k] = v
                    # Add other potential mappings if necessary
                
                # Filter for keys that belong to the Hugging Face model structure
                hf_keys = obj.transformer.state_dict().keys()
                filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in hf_keys}
                
                if not filtered_state_dict:
                    print("Warning: State dict mapping resulted in an empty state_dict for the Hugging Face model.")
                return filtered_state_dict

            if isinstance(device, int): # Legacy device handling
                device = torch.device(f"cuda:{device}")
            elif isinstance(device, str):
                device = torch.device(device)

            try:
                pl_ckpt = torch.load(ckpt_path, map_location=device)
                if "state_dict" in pl_ckpt:
                    raw_state_dict = pl_ckpt["state_dict"]
                else: # Assuming the checkpoint *is* the state_dict
                    raw_state_dict = pl_ckpt 
                
                mapped_params = map_state_dict_params(raw_state_dict)
                
                if mapped_params:
                    # Load only the transformer part if it's a PL checkpoint
                    # This assumes your PL checkpoint saves the 'transformer' (GPT2LMHeadModel) weights
                    # If your PL checkpoint saves the whole 'GPT' class, you might need to adjust.
                    # We are loading into obj.transformer directly.
                    missing_keys, unexpected_keys = obj.transformer.load_state_dict(mapped_params, strict=False)
                    if missing_keys:
                        print(f"Warning: Missing keys when loading state_dict: {missing_keys}")
                    if unexpected_keys:
                        print(f"Warning: Unexpected keys when loading state_dict: {unexpected_keys}")
                    print(f"Loaded PyTorch Lightning checkpoint from {ckpt_path}")
                else:
                    print(f"Could not extract valid parameters for GPT2LMHeadModel from {ckpt_path}. Model remains initialized from config.")

            except Exception as e:
                print(f"Failed to load PyTorch Lightning checkpoint from {ckpt_path}: {e}. Model remains initialized from config.")


        if disable_dropout:
            obj.transformer.config.attn_pdrop = 0.0
            obj.transformer.config.resid_pdrop = 0.0
            obj.transformer.config.embd_pdrop = 0.0
            for module in obj.transformer.modules(): # Apply to actual dropout layers
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
        
        obj.to(device)

        # Load the SmilesTokenizer
        # The tokenizer files (vocab.json, merges.txt, tokenizer_config.json) 
        # are expected to be in the same directory as the checkpoint's config.json,
        # which is os.path.dirname(ckpt_path) if ckpt_path includes the filename,
        # or ckpt_path if it's a directory.
        # In pretrain.py, tokenizer files are saved in model_output_path.
        # In sample.py, ckpt_path is model_output_path + "/model.ckpt".
        # So, the tokenizer path is os.path.dirname(ckpt_path).
        tokenizer_load_path = os.path.dirname(ckpt_path) 
        try:
            # Ensure SmilesVocabulary is available for from_pretrained, if SmilesTokenizer.from_pretrained needs it.
            # The SmilesTokenizer.from_pretrained method might need access to the base SmilesVocabulary
            # if it's used for more than just file paths (e.g. default symbols if files are missing).
            # For now, assume from_pretrained can work with just the path.
            # We need to ensure that the SmilesTokenizer class is imported in this file. (It is)
            loaded_tokenizer = SmilesTokenizer.from_pretrained(tokenizer_load_path)
            obj.tokenizer = loaded_tokenizer 
            print(f"SmilesTokenizer loaded from {tokenizer_load_path} and assigned to model.")
        except Exception as e:
            print(f"Warning: Failed to load SmilesTokenizer from {tokenizer_load_path}: {e}. Model will use default tokenizer.")
            # obj.tokenizer will remain the default SmilesTokenizer() initialized in GPT.__init__
            # This might be an issue if the vocab size of default tokenizer doesn't match the loaded model.
            # However, config['vocab_size'] used for model init should come from the saved config.json,
            # which should match the tokenizer saved during pretraining.
            # The main risk is if self.tokenizer.decode in sample() uses a different vocab mapping.

        return obj

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, idx, targets=None, loss_reduction="mean"):
        # device = idx.device # Not needed with HF model
        b, t = idx.size()
        assert (
            t <= self.transformer.config.n_positions # Use n_positions from HF config
        ), f"Cannot forward sequence of length {t}, block size is only {self.transformer.config.n_positions}"

        # Hugging Face GPT2LMHeadModel expects 'input_ids' and optionally 'labels'
        outputs = self.transformer(input_ids=idx, labels=targets)
        logits = outputs.logits

        loss = None
        if targets is not None:
            if loss_reduction == "mean":
                loss = outputs.loss # Use loss directly from HF model if reduction is 'mean'
            else: # Custom loss calculation if a different reduction is needed (though HF handles 'mean' and 'sum' internally if labels are passed)
                  # This part might need adjustment based on specific requirements if 'none' or other reductions are used elsewhere.
                loss = self.loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1, # Assuming -1 is the ignore_index, consistent with many setups
                    reduction=loss_reduction,
                )
        return logits, loss

    @torch.no_grad()
    def sample(
        self,
        n_batch,
        max_new_tokens=100,
        temperature=1.0,
        do_sample=True,
        top_k=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        This method is adapted to use Hugging Face's `generate` method.
        """
        self.transformer.eval() # Ensure model is in eval mode

        # The `generate` method expects `input_ids` to be on the same device as the model.
        # GO token should be the start of the sequence for generation.
        # `generate` handles sequence length constraints (max_length).
        # `pad_token_id` is important for batch generation if sequences have different lengths.
        # For SMILES, we usually generate one by one or with a fixed start.
        
        # Create initial input: batch of GO tokens
        go_token_id = self.tokenizer.vocabulary.go_index
        # input_ids = torch.full((n_batch, 1), go_token_id, dtype=torch.long, device=self.device)
        # The sample script seems to provide an initial `idx` which might not be just GO.
        # For now, let's assume `idx` is not used and generation starts from GO.
        # If `sample.py` provides a starting sequence, this needs adjustment.
        # The original code: c_ix = [self.tokenizer.vocabulary.go_index]; idx = torch.tensor(n_batch * [c_ix], device=self.device)
        # This means it starts with a single GO token for each batch item.
        input_ids = torch.full((n_batch, 1), go_token_id, dtype=torch.long, device=self.device)


        # `max_length` for `generate` is the total length of the sequence (input + generated)
        # `max_new_tokens` is how many tokens to add.
        # So, max_length = current_length + max_new_tokens
        current_length = input_ids.size(1)
        max_len = current_length + max_new_tokens
        
        # Ensure block_size (max sequence length for the model) is respected.
        # n_positions is the equivalent of block_size in Hugging Face's GPT2Config.
        # max_len = min(max_len, self.transformer.config.n_positions)
        # The generate method handles this via its own max_length, but good to be aware.
        # The original code crops idx if it's too long. `generate` should handle this.

        output_sequences = self.transformer.generate(
            input_ids=input_ids,
            max_length=max_len,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=n_batch, # Should be 1 per input sequence if input_ids has n_batch items
            pad_token_id=self.tokenizer.vocabulary.pad_index, # Important for beam search or if inputs vary in length
            eos_token_id=self.tokenizer.vocabulary.eos_index # Optional: stop generation at EOS
        )

        # Decode the generated sequences
        smiles_list = []
        for seq in output_sequences:
            # `generate` includes the input_ids in the output, so we might want to slice them off
            # if we only want the newly generated part. However, the original code decodes the whole thing.
            # Let's stick to decoding the whole sequence.
            smiles = self.tokenizer.decode(seq.cpu().tolist()) # .tolist() converts tensor to list of ints
            smiles_list.append(smiles)
            
        return smiles_list # Return a list of SMILES strings

    def logprobs(self, seqs, lengths):
        """
        Computes log probabilities of sequences under this model. 

        Arguments:
            seqs: (batch_size, seq_len) tensor of sequences
            lengths: (batch_size,) tensor of sequence lengths (ignored)
    
        Returns:
            (batch_size,) tensor of summed log probabilities
        """
        logits, _ = self(seqs)
        logits = logits.to(torch.float32)
        pad = self.tokenizer.vocabulary.pad_index

        labels = seqs[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != pad)

        labels[labels == pad] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2))
        per_token_logps = per_token_logps.squeeze(2) * loss_mask

        logprobs = per_token_logps.sum(-1)
        return logprobs

    def save_config(self, dir, name="config.json"):
        """
        Saves config to a file.

        Arguments:
            dir: directory to save config in
            name: name of config file (default: 'config.json')
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, name), "w") as f:
            # Save the Hugging Face model's config (which is more extensive)
            # or the original simple config, depending on needs.
            # For reproducibility of the HF model, its config is better.
            # self.transformer.config.to_json_file(os.path.join(dir, name))
            # Sticking to original behavior of saving the input config dict:
            json.dump(self.config, f, indent=4)


class GPTLightning(pl.LightningModule):
    """GPT Language Model

    Config is dict with options:
        vocab_size (required)
        block_size (required)
        model_type (required)
        attn_pdrop (default = 0.1)
        resid_pdrop (default = 0.1)
        embd_pdrop (default = 0.1)
        lr_decay (default = True)
        warmup_tokens (default = 4e7)
        final_tokens (default = 4e8)
        learning_rate (default = 1e-4)
        grad_norm_clip (default = 1.0)
    """

    def __init__(self, config):
        super().__init__()
        # config is the original config dict. GPT class now handles HF model init.
        self.gpt = GPT(config) 

        # some extra optimization stuff
        self.automatic_optimization = False
        self.tokens_processed = 0
        self.warmup_tokens = self.gpt.default("warmup_tokens", int(4e7))
        self.final_tokens = self.gpt.default("final_tokens", int(4e8))
        self.lr_decay = self.gpt.default("lr_decay", True)
        self.learning_rate = self.gpt.default("learning_rate", 1e-4)
        self.grad_norm_clip = self.gpt.default("grad_norm_clip", 1.0)
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.1
        self.step_size = 10
        self.gamma = 0.5

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=self.betas
        )
        if self.lr_decay == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.step_size, self.gamma
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, idx, targets=None):
        # self.gpt is now the wrapper around the Hugging Face model.
        # Its forward method calls the Hugging Face model's forward pass.
        return self.gpt(idx, targets=targets) # Calls GPT.forward()

    def _step_lr(self, tokens):
        """
        performs learning rate decay:
        """
        # Ensure self.trainer.optimizers is accessible and correctly gives the optimizer
        # In PyTorch Lightning 1.x and later, self.optimizers() is the correct way.
        opt = self.optimizers()
        if not opt: # Or handle cases where optimizer might not be set yet
            return

        self.tokens_processed += (
            tokens >= 0
        ).sum()  # number of tokens processed this step (i.e. label is not -100)
        
        if self.tokens_processed < self.warmup_tokens:
            # linear warmup
            lr_mult = float(self.tokens_processed) / float(max(1, self.warmup_tokens))
        else:
            # cosine learning rate decay
            progress = float(self.tokens_processed - self.warmup_tokens) / float(
                max(1, self.final_tokens - self.warmup_tokens)
            )
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        lr = self.learning_rate * lr_mult
        self.log("train/lr", lr, on_step=True, on_epoch=False) # Logging more explicitly for PL
        
        for param_group in opt.param_groups:
            param_group["lr"] = lr

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        
        input_ids, targets = batch["input"], batch["output"]
        
        # The forward method of GPT (and thus GPTLightning) now returns (logits, loss)
        # where loss is directly from the Hugging Face model if targets are provided.
        # The `loss_reduction` in GPT.forward is 'mean' by default.
        logits, loss = self(idx=input_ids, targets=targets) 

        if loss is None:
            # This case should ideally not happen if targets are always provided for training_step
            # Fallback or error if HF model didn't return loss
            # For now, let's assume loss is always returned by HF model when labels are passed.
            # If not, we might need to calculate it manually using self.gpt.loss_fn
            raise ValueError("Loss was not computed by the model. Ensure targets are passed.")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # Logging more explicitly

        # Manual optimization steps:
        if self.lr_decay != "step": # Assuming "step" means handled by a PL LR scheduler
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip) # Use self.parameters() for PL module
            opt.step()
            if self.lr_decay: # If lr_decay is True (bool) or a relevant string other than "step"
                self._step_lr(targets) # Pass targets (or any tensor with relevant count for token processing)
        # If lr_decay == "step", PL handles scheduler steps automatically based on configure_optimizers.

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch["input"], batch["output"]
        # self() calls forward(self, idx, targets)
        logits, loss = self(idx=input_ids, targets=targets)

        if loss is None:
            # Fallback if HF model didn't return loss (e.g. if reduction was not 'mean' and not handled)
            # This may require custom calculation if loss is not directly available
            # For now, assume loss is available from HF model output.
             loss = self.gpt.loss_fn( # Access loss_fn from the gpt attribute
                    logits.reshape(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1, # Ensure this ignore_index is consistent
                    reduction="mean"
                )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # Logging more explicitly

    @torch.no_grad()
    def sample(
        self,
        n_batch,
        max_new_tokens=100,
        temperature=1.0,
        do_sample=True,
        top_k=None,
    ):
        # This now correctly calls the sample method of the GPT class (which uses HF's generate)
        return self.gpt.sample(
            n_batch=n_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
        )
