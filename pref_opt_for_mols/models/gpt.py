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
from ..dataset import SmilesTokenizer


# -----------------------------------------------------------------------------

GPT_MODEL_FACTORY = {
    "gpt-small": dict(n_layer=8, n_head=8, n_embd=256),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}

# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd, n_head, attn_pdrop, resid_pdrop, block_size
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


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
        self.attn_pdrop = self.default("attn_pdrop", 0.1)
        self.resid_pdrop = self.default("resid_pdrop", 0.1)
        self.embd_pdrop = self.default("embd_pdrop", 0.1)

        self.tokenizer = SmilesTokenizer()

        self.transformer = nn.ModuleDict(
            dict(
                type_emb=nn.Embedding(1, self.n_embd),
                tok_emb=nn.Embedding(self.vocab_size, self.n_embd),
                pos_emb=nn.Embedding(self.block_size, self.n_embd),
                drop=nn.Dropout(self.embd_pdrop),
                h=nn.ModuleList(
                    [
                        Block(
                            self.n_embd,
                            self.n_head,
                            self.attn_pdrop,
                            self.resid_pdrop,
                            self.block_size,
                        )
                        for _ in range(self.n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(self.n_embd),
            )
        )
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
                )

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        # n_params = sum(p.numel() for p in self.transformer.parameters())
        # print("number of parameters: %.2fM" % (n_params / 1e6,))
        self.loss_fn = F.cross_entropy

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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
    def load_from_checkpoint(cls, config, ckpt_path, device=0, disable_dropout=False):
        def map_state_dict_params(state_dict):
            try:
                return {k.split("gpt.")[1]: v for k, v in state_dict.items()}
            except IndexError:
                return state_dict

        obj = cls(config)
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        
        if disable_dropout:
            for module in obj.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
        
        params = torch.load(ckpt_path, map_location=device)["state_dict"]
        obj.load_state_dict(map_state_dict_params(params))
        obj.to(device)

        return obj

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, idx, targets=None, loss_reduction="mean"):
        # device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # shape (1, t)
        # forward the GPT model itself
        tok_emb = self.transformer.tok_emb(
            idx
        )  # token embeddings of shape (b, t, n_embd)
        # using type_as(idx) allows us to get around PL not automatically transfering the
        # device of tensors created after init ()
        pos_emb = self.transformer.pos_emb(
            torch.arange(0, t, dtype=torch.long).unsqueeze(0).type_as(idx)
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
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
        """
        c_ix = [self.tokenizer.vocabulary.go_index]
        idx = torch.tensor(n_batch * [c_ix], device=self.device)
        self.eval()

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        smiles = self.tokenizer.decode(idx.cpu())
        return smiles
    
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
        return self.gpt.forward(idx, targets)

    def _step_lr(self, tokens):
        """
        performs learning rate decay:
        """
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
        self.log("train/lr", lr)
        opt = self.optimizers()
        for param_group in opt.param_groups:
            param_group["lr"] = lr

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        input, output = batch["input"], batch["output"]
        logits, loss = self(idx=input, targets=output)
        self.log("train_loss", loss, prog_bar=True)
        if self.lr_decay != "step":
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip)
            opt.step()
            if self.lr_decay:
                self._step_lr(output)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input, output = batch["input"], batch["output"]
        _, loss = self(idx=input, targets=output)
        self.log("val_loss", loss, prog_bar=True)

    @torch.no_grad()
    def sample(
        self,
        n_batch,
        max_new_tokens=100,
        temperature=1.0,
        do_sample=True,
        top_k=None,
    ):
        return self.gpt.sample(n_batch, max_new_tokens, temperature, do_sample, top_k)
