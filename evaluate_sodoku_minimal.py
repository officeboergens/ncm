"""Solve one hard-coded Sudoku with the released hard-Sudoku NCM checkpoint.

This is a self-contained minimal example: it does not use evaluate.py or the
shard/merge scripts. It builds the Sudoku tokenizer, the frozen VQ-VAE and the
next-concept generator, then decodes one puzzle at every level of the
coarse-to-fine hierarchy and reports residuals against the known solution.

Run:
    python evaluate_sodoku_minimal.py
"""
import os

import torch
from omegaconf import OmegaConf

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from utils.misc import default_device

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT, "artifacts", "sudoku-hard")
CONFIG_PATH = os.path.join(ROOT, "config", "eval", "sudoku-hard.yaml")

# One hard Sudoku, "." marks an empty cell. This is the first puzzle from the
# hard test set, which the released checkpoint solves.
PUZZLE = [
    ".216.7.85",
    "76....2..",
    "..3..27..",
    ".523.....",
    ".........",
    "8.67.....",
    "2...84.36",
    "6.81..4..",
    ".4..9...7",
]

# Known solution (the matching reference from data/sudoku-hard/solutions.pickle).
# Encoding it with the frozen VQ-VAE gives the true latent z, so the script can
# report the actual residual z - z_hat[i] at every level instead of estimating
# it from the final generated level.
SOLUTION = [
    "921647385",
    "764835291",
    "583912764",
    "452369178",
    "179458623",
    "836721549",
    "217584936",
    "698173452",
    "345296817",
]


def encode_puzzle(tokenizer, rows):
    """Turn 9 rows of text into the model's 91-token prefix.

    The training format is ``<BOS> row|row|...|row <BOS>``: a leading BOS, the
    nine rows separated by the ``|`` token, and a trailing BOS. Blank cells are
    ``_`` in the tokenizer's vocabulary.
    """
    text = "<BOS>" + "|".join(rows).replace(".", "_") + "<BOS>"
    print("Model-Sodoku-Encoding:\n", text)
    tokenizer_encoded = tokenizer.encode(text)
    print("tokenizer.pre_tokenizer encoded:\n", tokenizer_encoded)
    return tokenizer_encoded


def encode_solution(tokenizer, rows, device):
    """Encoder latent of the known solution.

    The VQ-VAE encodes the 89-token solved board (81 cells + 8 separators).
    This gives the true latent ``z`` used by the residual quantizer, which is
    available at runtime because the solution is known.
    """
    ids = tokenizer.pre_tokenizer.encode("|".join(rows))
    expected = tokenizer.vqvae.context_length
    if len(ids) != expected:
        raise ValueError(f"solution encodes to {len(ids)} tokens, expected {expected}")
    return tokenizer.vqvae.encode_z(torch.tensor([ids], device=device))


def print_grid(title, rows):
    print(title)
    for r, row in enumerate(rows):
        if r in (3, 6):
            print("  " + "-" * 17)
        print("  " + " ".join(row))


def grid_rows(text):
    """Split decoded Sudoku text into nine rows, ignoring the '|' separators.

    Intermediate hierarchy levels may put separators in odd places; the cell
    sequence is always 81 symbols, so chunking it directly is more robust.
    """
    cells = text.replace("|", "")
    if len(cells) != 81:
        return None
    return [cells[i:i + 9] for i in range(0, 81, 9)]


def rms(t):
    """Root-mean-square over all latent dimensions (a scale-free size measure)."""
    return t.pow(2).mean().sqrt().item()


def residual_report(level, z_hats, z_true):
    """Print the residual bookkeeping for one level.

    Residuals live in latent space, not in Sudoku space:

    - ``z_hats[i] - z_hats[i-1]`` is the correction level ``i`` contributed;
    - ``z_hats[-1] - z_hats[i]`` estimates ``z - z_hats[i]``, the part of the
      encoder latent that levels ``i+1...`` still have to encode. This is the
      residual the multiscale residual quantizer minimizes during training
      (``z_rest`` in ``models/quant.py``). It is only known after all levels
      have run, so the final reconstruction stands in for the encoder latent.
    - ``z_true - z_hats[i]`` is the *true* residual: the actual latent error
      against the known solution. ``z_true`` is obtained by encoding the
      solution, so this is known at runtime without running finer levels.
    """
    z = z_hats[level]
    prev = z_hats[level - 1] if level > 0 else torch.zeros_like(z)
    addition = z - prev
    remaining = z_hats[-1] - z
    frac = rms(remaining) / max(rms(z_hats[-1]), 1e-12)
    if level == len(z_hats) - 1:
        cos = "     -"
    else:
        cos = f"{torch.nn.functional.cosine_similarity(addition.reshape(-1), remaining.reshape(-1), dim=0).item():+.3f}"
    print(f"  residual: added RMS={rms(addition):.4f}  remaining RMS={rms(remaining):.4f} "
          f"({100 * frac:4.1f}% of final)  cos(add, remaining)={cos}")

    true_residual = z_true - z
    true_frac = rms(true_residual) / max(rms(z_true), 1e-12)
    true_cos = torch.nn.functional.cosine_similarity(
        addition.reshape(-1), true_residual.reshape(-1), dim=0).item()
    print(f"  true:     residual RMS={rms(true_residual):.4f} "
          f"({100 * true_frac:4.1f}% of z)  cos(add, true)={true_cos:+.3f}")


def residual_conv_ratio(vq, level):
    """Mixing ratio of the per-level residual convolution (`Phi`), if enabled.

    This is the *other* residual in the code: after upsampling a level's codes,
    ``Phi`` blends the upsampled vector with a 3-tap convolution of it,
    ``(1-r)*h + r*conv(h)`` (see ``Phi`` in ``models/quant.py``).
    """
    if not getattr(vq, "quant_resi_enabled", False):
        return None
    at = level / max(len(vq.levels) - 1, 1)
    ratio = getattr(vq.quant_resi[at], "resi_ratio", None)
    return None if ratio is None else float(ratio)


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    device = default_device()

    # The checkpoint filenames in the config are relative to their directory.
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(CHECKPOINT_DIR, cfg.checkpoints.tokenizer)

    # 1. Tokenizer = Sudoku pre-tokenizer + frozen multiscale VQ-VAE.
    tokenizer = VQVAETokenizer.from_config(cfg.tokenizer)
    tokenizer.vqvae = tokenizer.vqvae.to(device).eval()

    # 2. Generator, conditioned on the frozen tokenizer.
    net = NextConceptTransformer(vqvae_local=tokenizer.vqvae, config=cfg.model).to(device)
    state = torch.load(os.path.join(CHECKPOINT_DIR, cfg.checkpoints.generator),
                       map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"], strict=True)
    net.eval()

    # 3. Encode the puzzle as the prefix the generator conditions on.
    prefix_ids = encode_puzzle(tokenizer.pre_tokenizer, PUZZLE)
    x = torch.tensor([prefix_ids], device=device)

    # 4. Generate the code hierarchy coarse to fine. `z_hats[i]` is the
    #    cumulative full-length latent after level i, so decoding each one
    #    shows the model's answer at that stage of refinement.
    with torch.no_grad():
        codes = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
        z_hats, _ = tokenizer.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        z_true = encode_solution(tokenizer, SOLUTION, device)
        level_token_ids = [
            tokenizer.vqvae.decoder(z).argmax(-1)[0].tolist()
            for z in z_hats
        ]

    print()
    print_grid("Puzzle:", PUZZLE)

    # 5. Decode and print the answer produced at every hierarchy level.
    levels = tokenizer.vqvae.vq.levels
    print("\nResiduals are measured on the 89x256 latent:")
    print("  added     = what this level contributed (z_hat[i] - z_hat[i-1])")
    print("  remaining = what the finer levels still have to encode (z_hat[last] - z_hat[i])")
    print("  true      = actual error against the known solution (z - z_hat[i], z = encoder(SOLUTION))")
    print("  phi       = residual-convolution mixing ratio at this level")
    for level, token_ids in enumerate(level_token_ids):
        text = tokenizer.pre_tokenizer.decode(token_ids, skip_special_tokens=True)
        rows = grid_rows(text)
        n_codes = levels[level]
        title = f"Level {level} ({n_codes} code{'s' if n_codes != 1 else ''}):"
        print()
        if rows is None:
            # Keep the raw output visible if a level did not decode to 81 cells.
            print(f"{title} {text}")
        else:
            print_grid(title, rows)
        residual_report(level, z_hats, z_true)
        phi = residual_conv_ratio(tokenizer.vqvae.vq, level)
        if phi is not None:
            print(f"  residual conv: phi={phi:.3f}")


if __name__ == "__main__":
    main()
