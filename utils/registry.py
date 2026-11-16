"""String-name -> class registries used by the YAML configs.

Configs refer to implementations by short names (``pre_tokenizer.name``,
``vector_quantizer_config.clss``); ``models/vqvae.py`` and
``tokenization/vqvae_tokenizer.py`` resolve them through these tables with
``hydra.utils.get_class``.
"""
import math
from utils.misc import round_levels

# Pre-tokenizers: text <-> pre-token ids, wrapped by VQVAETokenizer.
tokenizers = {
    "bpe": "tokenization.bpe_tokenizer.BPETokenizer",
    "hf": "tokenization.hf_tokenizer.HFTokenizer",
    "sudoku": "tokenization.sudoku_tokenizer.SudokuTokenizer",
    "vqvae": "tokenization.vqvae_tokenizer.VQVAETokenizer",
}

# Quantizer that turns encoder latents into the multiscale code hierarchy.
vector_quantizers = {
    "multiscale_residual_vector_quantizer": "models.quant.MultiscaleResidualVectorQuantizer",
}

# Optional schedules for choosing the sequence length of each hierarchy level
# (used when a config sets `levels.use_manual_levels: false`). Each returns a
# strictly increasing integer list from `start` to `end`.
level_schedulers = {
    "linear": lambda start, end, n: round_levels([start + (end - start) * (i-1)/(n-1) for i in range(1, n+1)], start, end),
    "cubic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**3/(n-1)**3 for i in range(1, n+1)], start, end),
    "exp": lambda start, end, n: round_levels([start + (end - start) * (math.exp(i-1))/(math.exp(n-1)) for i in range(1, n+1)], start, end),
    "sqrt": lambda start, end, n: round_levels([start + (end - start) * math.sqrt(i-1)/math.sqrt(n-1) for i in range(1, n+1)], start, end),
    "sine": lambda start, end, n: round_levels([start + (end - start) * math.sin((math.pi/2) * ((i-1)/(n-1))) for i in range(1, n+1)], start, end),
    "log": lambda start, end, n: round_levels([start + (end - start) * math.log(i)/math.log(n) for i in range(1, n+1)], start, end),
    "quadratic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**2/(n-2)**2 for i in range(1, n+1)], start, end),
}
