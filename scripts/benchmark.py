import os
import time
import argparse
import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from pref_opt_for_mols.metrics import internal_diversity, frac_unique, frac_valid, fcd_score
from pref_opt_for_mols.filter import filter_mcf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gen",
    nargs="+",
    type=str,
    help="smiles .csv files generated with sample.py",
)
parser.add_argument(
    "--recompute",
    action="store_true",
    help="recompute the benchmark results even if already cached",
)
parser.add_argument(
    "--ref_smiles",
    type=str,
    help="reference smiles to compute fcd score against"
)
parser.add_argument(
    "--metrics",
    nargs="+",
    type=str,
    default=["frac_valid", "frac_unique", "internal_diversity", "fcd_score", "mcf"],
    help="metrics to compute"
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="device to run fcd benchmark on"
)
parser.add_argument(
    "--dump",
    type=str,
    help="benchmark directory to save metrics/plot to"
)
args = parser.parse_args()


def fn_name(fn):
    try:
        return fn.__name__
    except AttributeError:
        return fn.func.__name__


def mcf(smiles):
    return np.mean(filter_mcf(smiles, n_jobs=64))


if __name__ == "__main__":
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
 
    to_compute = []
    metrics = {
        "frac_valid": partial(frac_valid, n_workers=16),
        "frac_unique": partial(frac_unique, n_workers=16),
        "internal_diversity": partial(internal_diversity, n_workers=16),
        "mcf": mcf
    }
    if args.ref_smiles:
        metrics["fcd_score"] = partial(
            fcd_score,
            ref_smiles=pd.read_csv(args.ref_smiles)["smiles"].tolist(),
            device=f"cuda:{args.device}",
            n_workers=16
        )
    else:
        print("Warning: no reference smiles provided, skipping fcd_score")
    
    retrieved = {}
    selected_metrics = [metrics[k] for k in args.metrics]
    print(f"Selected metrics: {[fn_name(m) for m in selected_metrics]}")

    for genfile in args.gen:
        name = os.path.basename(genfile)

        if name in os.listdir(args.dump) and not args.recompute:
            to_continue = True

            retrieved[name] = pd.read_csv(os.path.join(args.dump, name))
            retrieved[name] = retrieved[name].set_index("name")

            computed = set(retrieved[name].columns.tolist())
            selected = set([fn_name(m) for m in selected_metrics])
            if len(selected - computed) > 0:
                to_continue = False
    
            if to_continue:
                print(f"Skipping {genfile} (already computed)")
                continue

        df = pd.read_csv(genfile)["smiles"].tolist()
        to_compute.append({
            "data": df,
            "name": name,
            "metrics": selected_metrics
        })

    print("Collected {} benchmarks to compute".format(len(to_compute)))

    benchmarked = {}

    for i, entry in enumerate(to_compute):
        s = time.time()
        print(f"({i + 1}/{len(to_compute)}) "
              f"Computing metrics for {entry['name']} "
              f"(n={len(entry['data'])})")

        results = {"name": entry["name"]}
        for metric in entry["metrics"]:
            result = metric(entry["data"])
            results[fn_name(metric)] = result
            print(f" - {fn_name(metric)}: {round(result, 3)}")
 
        name = ".".join(entry["name"].split(".")[:-1])
        benchmarked[name] = pd.DataFrame([results]).set_index("name")
        if args.dump:
            benchmarked[name].to_csv(os.path.join(args.dump, entry["name"]))

        print(f"({i + 1}/{len(to_compute)})", end=" ")
        print("Done in {:.2f}s".format(time.time() - s))

    benchmarked = {**benchmarked, **retrieved}
    benchmarked = pd.concat(list(benchmarked.values()), axis=0)
    df = benchmarked.transpose()

    ax = df.plot(kind="bar", figsize=(8, 6))

    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title(f"Generation benchmarking for {len(benchmarked)} datasets")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.legend(title='Names')

    plt.savefig(os.path.join(args.dump, "benchmark.png"), dpi=300)
    print(f"Saved benchmark plot to {args.dump}/benchmark.png")
