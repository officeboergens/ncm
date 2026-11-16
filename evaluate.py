"""Parallel evaluation entry point.

Typical invocation:

    python evaluate.py --config config/eval/xsum.yaml \
        --checkpoint-dir artifacts/xsum --data-dir data/xsum

The run happens in two phases:

1. generation: one ``scripts/eval_shard.py`` process per GPU. Each process
   generates the hypotheses for its slice of the inputs and pickles them to
   ``<out>/shard_<k>.pkl`` (stdout/stderr go to ``<out>/shard_<k>.log``).
2. scoring: ``scripts/eval_merge.py`` loads every shard pickle, restores the
   original input order and prints the task metric.

This file never loads a model; it only parses arguments, decides how to shard
and supervises the child processes.
"""
import argparse, os, subprocess, sys

import torch
from omegaconf import OmegaConf

# Absolute repo root. Children are started with ``cwd=ROOT`` so that the
# relative paths in configs ("artifacts/...", "data/...", "eval/...") resolve
# the same way regardless of where the user invoked the script from.
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="one generation shard per GPU, then merge and score")
    ap.add_argument("--config", required=True, help="eval config, e.g. config/eval/xsum.yaml")
    ap.add_argument("--checkpoint-dir", required=True, help="directory with the files named under checkpoints:")
    ap.add_argument("--data-dir", default="", help="directory with the files named under data: (not needed for unconditional tasks)")
    # Default output is ``eval/<task>`` (relative to the caller's cwd).
    ap.add_argument("--out", default="", help="output directory (default eval/<task>)")
    # Comma-separated physical GPU ids; empty means "all visible GPUs".
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n inputs")
    # Unknown flags are not rejected: they are forwarded verbatim to the shard
    # script (--temp, --top_k, --top_p, --cfg, --kv-cache, ...).
    args, passthrough = ap.parse_known_args()

    # The config is the task description: checkpoints, data files, model and
    # tokenizer dimensions, generation defaults and the eval metric.
    cfg = OmegaConf.load(args.config)

    # Decide the shard count. Sharding is driven by the GPU list, not by the
    # `eval.n_shards` key that also appears in the configs.
    gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(torch.cuda.device_count())]
    if not gpus:
        # no CUDA: run a single shard on the local device (MPS on Apple silicon, else CPU)
        gpus = [None]
    out = args.out or os.path.join("eval", cfg.task)
    os.makedirs(out, exist_ok=True)

    # Arguments shared by every shard. Each shard additionally gets its own
    # index and output file path (see below).
    common = ["--config", args.config, "--checkpoint-dir", args.checkpoint_dir] + (["--data-dir", args.data_dir] if args.data_dir else []) + [
              "--nshards", str(len(gpus))] + (["--n", str(args.n)] if args.n else []) + passthrough
    procs = []
    for k, gpu in enumerate(gpus):
        # Each child only sees the GPU it should use, so models land on the
        # intended device even though the config has no device setting.
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        # Keep one log per shard so a failing worker can be inspected alone.
        log = open(os.path.join(out, f"shard_{k}.log"), "w")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "scripts", "eval_shard.py"), *common,
             "--shard", str(k), "--out", os.path.join(out, f"shard_{k}.pkl")],
            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
    where = f"GPU(s) {','.join(gpus)}" if gpus != [None] else "local device (no CUDA)"
    print(f"[evaluate] {len(gpus)} shard(s) on {where} -> {out}/", flush=True)

    # Wait for every shard. A non-zero exit means at least one worker crashed;
    # point the user at the corresponding log instead of printing a traceback.
    failed = [k for k, p in enumerate(procs) if p.wait() != 0]
    if failed:
        sys.exit(f"shard(s) {failed} failed; see {out}/shard_*.log")

    # Merge the pickles (restoring input order) and compute the metric selected
    # by `eval.metric` in the config.
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "eval_merge.py"), "--config", args.config,
                    "--glob", os.path.join(out, "shard_*.pkl")], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
