import os
import time
import json
import shutil
from tqdm import tqdm
from collections import defaultdict
from neptune.types import File

import torch
from torch import nn
import torch.nn.functional as F


def dpo_loss(
    policy_pos_logprobs,
    policy_neg_logprobs,
    ref_pos_logprobs,
    ref_neg_logprobs,
    beta=0.5,
):
    """
    Computes DPO loss under human preference model given preferred/rejected
    log probabilities. See https://arxiv.org/pdf/2305.18290.pdf.

    Arguments:
        policy_pos_logprobs: log prob of preferred seqs under current model (batch_size,)
        policy_neg_logprobs: log prob of rejected seqs under current model (batch_size,)
        ref_pos_logprobs: log prob of preferred seqs under reference model (batch_size,)
        ref_neg_logprobs: log prob of rejected seqs under reference model (batch_size,)
        beta: temperature parameter (float scalar, default 0.5)

    Returns:
        loss, accepted, and rejected reward tensors (batch_size,)
    """
    policy_logratios = policy_pos_logprobs - policy_neg_logprobs
    ref_logratios = ref_pos_logprobs - ref_neg_logprobs

    loss = -F.logsigmoid(beta * (policy_logratios - ref_logratios))
    pos_rewards = beta * (policy_pos_logprobs - ref_pos_logprobs).detach()
    neg_rewards = beta * (policy_neg_logprobs - ref_neg_logprobs).detach()

    return loss, pos_rewards, neg_rewards


class DPO:
    """Direct preference optimization for gpt/rnn models."""

    def __init__(
        self,
        reference,
        policy,
        config,
        device="cuda:0",
        logger=None,
        run=None,
    ):
        """
        Initializes DPO trainer. Sensible defaults are provided for training hyperparams,
        minus the 'model_path', which must be specified in the initial config. This is
        where checkpoints will be stored.

        Arguments:
            reference: pretrained reference model
            policy: pretrained policy model
            config: dictionary of training hyperparameters
            device: device to use for training (str, default "cuda:0")
            logger: logger to use for training (type NeptuneLogger, default None)
            run: run to log training progress (type neptune.run.Run, default None)
        """
        self.config = config
        self.dump_path = config["model_path"]

        self.beta = self.default("beta", 0.5)
        self.lr = self.default("lr", {"stop": 5e-7, "steps": 150})
        self.grad_norm_clip = self.default("grad_norm_clip", 10.0)
        self.gradient_accumulation = self.default("gradient_accumulation", 2)
        self.min_log_interval = self.default("min_log_interval", 2.0)

        self.eval_every = self.default("eval_every", 20000)
        self.max_epochs = self.default("max_epochs", 5)

        self.policy = policy
        self.reference = reference

        self.reference = reference.half()
        self.reference.eval()

        self.device = torch.device(device)
        self.policy.to(self.device)
        self.reference.to(self.device)

        self.logger = logger
        self.run = run
        if run and not logger or logger and not run:
            raise ValueError("Must provide both logger and run, or neither")
        if self.logger:
            self.config["neptune"] = self.run.get_url()
            cfg = {k: (v if v is not None else "null") for k, v in self.config.items()}
            self.run[self.logger.base_namespace]["config"] = cfg

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

    def save_configs(self, dir, dpo_name="config-dpo.json", base_name="config.json"):
        """
        Saves base model and DPO training configs to directory.

        Arguments:
            dir: directory to save configs to (str)
            dpo_name: name of DPO config file (str, default "config-dpo.json")
            base_name: name of base config file (str, default "config.json")
        """
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, base_name), "w") as f:
            json.dump(self.policy.config, f, indent=4)
        with open(os.path.join(dir, dpo_name), "w") as f:
            json.dump(self.config, f, indent=4)

    def loss(self, batch, prefix="train"):
        """
        Computes DPO loss and other metrics for a batch of data.

        Arguments:
            batch: batch of data (type dict, keys 'positive', 'negative',
                'positive_length', 'negative_length')
            prefix: prefix to use for logging (str, default "train")

        Returns:
            loss, dict of metrics including acc, loss, margin, pos/neg rewards,
            pos/neg log probabilities
        """
        pos = batch["positive"].to(self.device)
        neg = batch["negative"].to(self.device)
        pos_len = batch["positive_length"].to(self.device)
        neg_len = batch["negative_length"].to(self.device)

        seqs = torch.cat([pos, neg])
        lengths = torch.cat([pos_len, neg_len])

        policy_logprobs = self.policy.logprobs(seqs, lengths)
        policy_pos, policy_neg = torch.split(policy_logprobs, pos.shape[0])

        with torch.no_grad():
            ref_logprobs = self.reference.logprobs(seqs, lengths)
            ref_pos, ref_neg = torch.split(ref_logprobs, pos.shape[0])

        loss, pos_reward, neg_reward = dpo_loss(
            policy_pos, policy_neg, ref_pos, ref_neg, beta=self.beta
        )

        acc = (pos_reward > neg_reward).float().mean()
        margin = (pos_reward - neg_reward).mean()
        pos_reward, neg_reward = pos_reward.mean(), neg_reward.mean()
        pos_logprobs, neg_logprobs = (
            policy_pos.detach().mean(),
            policy_neg.detach().mean(),
        )
        loss_val = loss.detach().mean()

        return loss.mean(), {
            f"{prefix}/acc": acc,
            f"{prefix}/loss": loss_val,
            f"{prefix}/margin": margin,
            f"{prefix}/pos_reward": pos_reward,
            f"{prefix}/neg_reward": neg_reward,
            f"{prefix}/pos_logprobs": pos_logprobs,
            f"{prefix}/neg_logprobs": neg_logprobs,
        }

    def log(self, metrics, examples):
        """
        Logs metrics to logger and run. Does nothing if no logger specified.

        Arguments:
            metrics: dictionary of metrics to log
            examples: number of examples seen so far (int)
        """
        if self.logger:
            for key, value in metrics.items():
                if not isinstance(value, File):
                    value = value.item()
                self.run[self.logger.base_namespace][key].append(
                    value=value, step=examples
                )

    def save_checkpoint(self, metrics=None, examples=None):
        """
        Saves policy only to checkpoint path.

        Arguments:
            metrics: dictionary of metrics to save (None)
            examples: number of examples seen so far (None)
        """
        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "metrics": metrics or {},
            "examples": examples or 0,
        }
        model_name = f"model-{examples}.ckpt" if examples is not None else "model.ckpt"
        torch.save(checkpoint, os.path.join(self.dump_path, model_name))
        if model_name != "model.ckpt":
            curr = os.path.join(self.dump_path, "model.ckpt")
            try:
                os.remove(curr)
            except FileNotFoundError:
                pass
            shutil.copyfile(os.path.join(self.dump_path, model_name), curr)

    def train(self, train, val=None):
        """
        Trains policy on training data, evaluating on validation data.
        Uses RMSprop with linear warmup/constant LR schedule, as per original paper.
        Support for model checkpointing, logging to Neptune, and gradient accumulation.
        This is preferred over a Pytorch Lightning implementation because
        logging/checkpointing is more convenient, and this is probably a little faster.

        Arguments:
            train: training dataloader (type torch.utils.data.DataLoader)
            val: validation dataloader (type torch.utils.data.DataLoader, default None)
        """

        def print_info(metrics, key):
            print(
                f"({key}) "
                f"epoch: {epoch} | "
                f"examples: {examples} | "
                f"step: {step} | "
                f"acc: {metrics['{}/acc'.format(key)]:.3f} | "
                f"pos_reward: {metrics['{}/pos_reward'.format(key)]:.3f} | "
                f"neg_reward: {metrics['{}/neg_reward'.format(key)]:.3f} | "
                f"elapsed: {time.time() - start_time:.2f}s"
            )

        optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=self.lr["stop"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1.0, (step + 1) / (self.lr["steps"] + 1))
        )

        self.policy.train()
        start_time = time.time()
        last_log = None
        examples = 0
        step = 0
        since_last_val = 0

        for epoch in range(1, self.max_epochs + 1):
            train_iter = iter(train)
            while True:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break

                # Evaluation step
                if val is not None and (
                    since_last_val >= self.eval_every or examples == 0
                ):
                    self.policy.eval()
                    eval_metrics = defaultdict(list)

                    with torch.no_grad():
                        for val_batch in tqdm(val, desc=f"eval @ {examples} examples"):
                            _, batch_metrics = self.loss(val_batch, prefix="eval")
                            for k, v in batch_metrics.items():
                                eval_metrics[k].append(v)

                    eval_metrics = {k: sum(v) / len(v) for k, v in eval_metrics.items()}
                    self.log(eval_metrics, examples=examples)
                    print_info(eval_metrics, "eval")

                    if examples > 0:
                        self.save_checkpoint(eval_metrics, examples)

                    since_last_val = 0
                    self.policy.train()

                # Compute loss and metrics
                metrics = defaultdict(list)
                batch_size = len(batch["positive"])
                chunk_size = batch_size // self.gradient_accumulation

                for idx in range(self.gradient_accumulation):
                    end = (idx + 1) * chunk_size
                    if batch_size - end < chunk_size:
                        end = batch_size
                    slicer = slice(idx * chunk_size, end)

                    microbatch = {k: v[slicer] for k, v in batch.items()}
                    loss, metrics_micro = self.loss(microbatch)
                    (loss / self.gradient_accumulation).backward()

                    for k, v in metrics_micro.items():
                        metrics[k].append(v)

                # Update parameters and scheduler
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.grad_norm_clip
                )
                metrics["train/norm"] = [norm]

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Log train metrics periodically
                if last_log is None or time.time() - last_log > self.min_log_interval:
                    last_log = time.time()
                    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

                    self.log(metrics, examples=examples)
                    print_info(metrics, "train")

                step += 1
                examples += len(batch["positive"])
                since_last_val += len(batch["positive"])

        print("done training, saving final checkpoint...")
        self.save_checkpoint(examples="final")
