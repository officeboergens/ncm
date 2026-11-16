"""Evaluate the hard Sudoku checkpoint. Takes no arguments: python evaluate_sodoku.py

This is a convenience wrapper around ``evaluate.py`` with the three paths
(config, checkpoints, data) already filled in for the released hard-Sudoku
model. Everything else (sharding, generation, scoring) happens there.
"""

import os
import subprocess
import sys

# Repo root; paths below are absolute so the wrapper works from any cwd.
ROOT = os.path.dirname(os.path.abspath(__file__))
TASK = "sudoku-hard"


def main():
    cmd = [
        sys.executable,
        os.path.join(ROOT, "evaluate.py"),
        # Task definition: model/tokenizer dims, generation settings, `metric: exact`.
        "--config", os.path.join(ROOT, "config", "eval", f"{TASK}.yaml"),
        # tokenizer.pt + generator.pt (and config.yaml) for this task.
        "--checkpoint-dir", os.path.join(ROOT, "artifacts", TASK),
        # puzzles.pickle (inputs) + solutions.pickle (references).
        "--data-dir", os.path.join(ROOT, "data", TASK),
    ]
    # Run as a subprocess (rather than importing evaluate) so the child gets a
    # clean interpreter and the exit code propagates unchanged.
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
