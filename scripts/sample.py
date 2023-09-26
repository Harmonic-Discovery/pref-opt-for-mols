import argparse
import pandas as pd
from pref_opt_for_mols.models import GPT, CharRNN
import torch
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arch",
    type=str,
    default="rnn",
    help="which model class to use, either 'gpt' or 'rnn'",
)
parser.add_argument(
    "--model_path",
    required=True,
    type=str,
)
parser.add_argument(
    "--num_batches",
    default=1,
    type=int,
)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
)
parser.add_argument(
    "--device",
    default=0,
    type=int,
)
parser.add_argument(
    "--out",
    required=True,
    type=str,
)
args = parser.parse_args()

if __name__ == "__main__":
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)

    if args.arch == "gpt":
        # when we load we make sure we first map the models to cpu,
        # and then transfer over to the desired device
        model = GPT.load_from_checkpoint(
            config,
            os.path.join(args.model_path, "model.ckpt"),
            device=args.device,
            disable_dropout=True,
        )
    elif args.arch == "rnn":
        model = CharRNN.load_from_checkpoint(
            config,
            os.path.join(args.model_path, "model.ckpt"),
            device=args.device,
            disable_dropout=True,
        )
    else:
        raise ValueError(f"Unrecognized model {args.arch}")

    device = torch.device(f"cuda:{args.device}")
    model.to(device)

    sampled_smiles = []
    for j in range(args.num_batches):
        curr_smiles = model.sample(n_batch=args.batch_size)
        sampled_smiles.extend(curr_smiles)
        print(
            f"{len(sampled_smiles)}/{args.batch_size*args.num_batches} SMILES sampled"
        )

    pd.DataFrame({"smiles": sampled_smiles}).to_csv(args.out)
