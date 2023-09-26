import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import pytorch_lightning as pl
from ..dataset import SmilesTokenizer


class CharRNN(nn.Module):
    def __init__(self, config):
        super(CharRNN, self).__init__()
        self.config = config

        self.tokenizer = SmilesTokenizer()
        self.hidden_size = config.get("hidden_size", 768)
        self.num_layers = config.get("num_layers", 3)
        self.dropout = config.get("dropout", 0.2)
        self.vocab_size = self.input_size = self.output_size = len(
            self.tokenizer.vocabulary
        )

        self.embedding_layer = nn.Embedding(
            self.vocab_size,
            self.vocab_size,
            padding_idx=self.tokenizer.vocabulary.pad_index,
        )
        self.lstm_layer = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

        self.loss_fn = nn.CrossEntropyLoss()

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
                return {k.split("rnn.")[1]: v for k, v in state_dict.items()}
            except IndexError:
                return state_dict

        if disable_dropout:
            # Cannot modify dropout post-instantiation because torch LSTM cell
            # uses optimized implementation that doesn't expose dropout module
            config["dropout"] = 0.0

        obj = cls(config)
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)

        params = torch.load(ckpt_path, map_location=device)["state_dict"]
        obj.load_state_dict(map_state_dict_params(params))
        obj.to(device)

        return obj

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, lengths, targets=None, hiddens=None):
        seq_len = x.shape[1]
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=self.tokenizer.vocabulary.pad_index,
            total_length=seq_len,
        )
        x = self.linear_layer(x)

        if targets is not None:
            loss = self.loss_fn(x.permute(0, 2, 1), targets)
            return x, lengths, hiddens, loss
        else:
            return x, lengths, hiddens

    def sample(self, n_batch, max_new_tokens=100):
        self.eval()
        with torch.no_grad():
            starts = [
                torch.tensor(
                    [self.tokenizer.vocabulary.go_index],
                    dtype=torch.long,
                    device=self.device,
                )
                for _ in range(n_batch)
            ]

            starts = torch.tensor(
                starts, dtype=torch.long, device=self.device
            ).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(
                    self.tokenizer.vocabulary.pad_index,
                    dtype=torch.long,
                    device=self.device,
                ).repeat(max_new_tokens + 2)
                for _ in range(n_batch)
            ]

            for i in range(n_batch):
                new_smiles_list[i][0] = self.tokenizer.vocabulary.go_index

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor(
                [1 for _ in range(n_batch)], dtype=torch.long, device=self.device
            ).cpu()
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            for i in range(1, max_new_tokens + 1):
                output, _, hiddens = self.forward(
                    x=starts, lengths=lens, hiddens=hiddens
                )

                # probabilities
                probs = [F.softmax(o, dim=-1) for o in output]

                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]

                for j, top in enumerate(ind_tops):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == self.tokenizer.vocabulary.eos_index:
                            end_smiles_list[j] = True

                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(
                    ind_tops, dtype=torch.long, device=self.device
                ).unsqueeze(1)

            new_smiles_list = [
                new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)
            ]
            return self.tokenizer.decode(new_smiles_list)

    def logprobs(self, seqs, lengths):
        """
        Computes log probabilities of sequences under this model.

        Arguments:
            seqs: (batch_size, seq_len) tensor of sequences
            lengths: (batch_size,) tensor of sequence lengths

        Returns:
            (batch_size,) tensor of summed log probabilities
        """
        logits, *_ = self(seqs, lengths.cpu())

        logits = logits.to(torch.float32)
        pad = self.tokenizer.vocabulary.pad_index

        labels = seqs[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != pad

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


class CharRNNLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.rnn = CharRNN(config)

        self.learning_rate = self.rnn.default("learning_rate", 1e-3)
        self.gamma = self.rnn.default("gamma", 0.5)
        self.step_size = self.rnn.default("step_size", 10)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            (p for p in self.rnn.parameters() if p.requires_grad), lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.step_size, self.gamma
        )
        return [optimizer], [scheduler]

    def forward(self, x, lengths, targets=None, hiddens=None):
        return self.rnn.forward(x, lengths, targets, hiddens)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        input, output, length = batch["input"], batch["output"], batch["length"].cpu()
        _, _, _, loss = self(x=input, targets=output, lengths=length)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input, output, length = batch["input"], batch["output"], batch["length"].cpu()
        _, _, _, loss = self(x=input, targets=output, lengths=length)
        self.log("val_loss", loss, prog_bar=True)

    def sample(self, n_batch, max_new_tokens=100):
        return self.rnn.sample(n_batch, max_new_tokens)
