"""Generate the hypotheses for ONE shard of an evaluation run.

``evaluate.py`` starts one process per GPU (or one on CPU/MPS) with
``--shard k --nshards N``. Every process:

1. loads the eval config, the frozen VQ-VAE tokenizer and the generator;
2. selects the inputs (conditional tasks) or random seeds (unconditional
   tasks) that belong to its shard;
3. runs coarse-to-fine generation and decodes the codes back to text;
4. pickles ``{"idxs", "hyps", "refs"}`` plus extra keys for unconditional
   tasks, for ``eval_merge.py`` to merge and score.

The shard pickles are the contract with ``eval_merge.py``: ``idxs`` records
the original dataset position of every hypothesis so shard order does not
matter when merging.
"""
import argparse, os, pickle, sys
import torch
from omegaconf import OmegaConf

# Allow imports when the process is started from outside the repo root.
# evaluate.py already sets cwd=ROOT; NCM_ROOT can override for custom setups.
sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer
from utils.misc import default_device, wide_float

ap = argparse.ArgumentParser()
# Paths and sharding, supplied by evaluate.py.
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", default="")
ap.add_argument("--data-dir", default="")
ap.add_argument("--split", default="")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--bs", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
# Optional overrides for the `generation:` block in the config.
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--top_k", type=int, default=None)
ap.add_argument("--top_p", type=float, default=None)
ap.add_argument("--cfg", type=float, default=None)
ap.add_argument("--freq", type=float, default=None)
ap.add_argument("--mu", type=float, default=None)
ap.add_argument("--adj", default=None)
ap.add_argument("--l0", default="", help="unigram (default), active, uniform, or smooth:<eps> (mix eps of uniform into the unigram)")
ap.add_argument("--out", required=True)
ap.add_argument("--kv-cache", action="store_true", help="encode the prefix once and reuse its keys/values at every level")
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
# NCM_DEVICE > CUDA > Apple MPS > CPU. ``F64`` is used for the count-based
# sampling penalties; MPS has no float64, so it falls back to float32 there.
DEV = default_device()
F64 = wide_float(DEV)
G = CFG.generation  # generation defaults from the YAML

# Checkpoint/data directories: CLI wins, otherwise use the conventional paths.
ck = a.checkpoint_dir or os.path.join("artifacts", CFG.task)
dd = a.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), CFG.task)

# Precedence for every generation knob: CLI flag > `generation:` in the YAML >
# the hardcoded default in the last position.
pick = lambda cli, key, cast, default: cast(cli if cli is not None else G.get(key, default))
temp = pick(a.temp, "temperature", float, 1.0)
top_k = pick(a.top_k, "top_k", int, 1)
top_p = pick(a.top_p, "top_p", float, 0.0)
cfg = pick(a.cfg, "cfg_scale", float, 0.0)
freq = pick(a.freq, "frequency_penalty", float, 0.0)
mu = pick(a.mu, "frequency_deficit", float, 0.0)
adj = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")
split = a.split or CFG.eval.get("split", "test")
n = a.n or int(CFG.eval.get("n_docs", 0))
bs = a.bs or int(CFG.eval.batch_size)

# Config paths are relative to the checkpoint directory.
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (('vocab_file_path', 'bpe_vocab'), ('merges_file_path', 'bpe_merges')):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])

# The tokenizer bundle is a pre-tokenizer (HF GPT-2, learned BPE, Sudoku, ...)
# plus the frozen multiscale VQ-VAE that turns pre-tokens into a hierarchy of
# discrete codes.
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(DEV).eval()

# The generator consumes the frozen VQ-VAE (passed as a tuple internally so it
# stays out of the generator's state dict) and predicts the code hierarchy.
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(DEV)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
P = int(CFG.model.prefix_len)  # number of prefix positions the model conditions on
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id  # may be None


def load_allowance(spec):
    """Load an auxiliary tensor named by a `generation.*` config entry.

    ``spec`` is either a filename inside the checkpoint directory or
    ``"file:key.path.inside.pickle"`` (e.g. ``generator.pt:inference.level0_unigram``).
    The tensors are moved to the device in float64 because they are used for
    count bookkeeping during sampling.
    """
    if ":" in spec:
        path, ref = spec.split(":", 1)
        obj = torch.load(os.path.join(ck, path), map_location="cpu", weights_only=False)
        for k in ref.split("."):
            obj = obj[k]
        return obj.to(DEV, F64)
    return torch.load(os.path.join(ck, spec), map_location="cpu").to(DEV, F64)


# Optional token-frequency controls from the original NCM sampler:
# - ALLOW: per-token allowance used by the frequency penalty/deficit;
# - L0_P:  level-0 unigram used to sample the coarsest code in unconditional runs.
ALLOW = load_allowance(str(G.frequency_allowance)) if G.get("frequency_allowance") else None
L0_MODE = a.l0 or "unigram"
L0_P = load_allowance(str(G.level0_unigram)).float() if G.get("level0_unigram") else None
if L0_P is not None and L0_MODE == "active":
    # Keep only codes observed in the reference distribution and renormalize.
    L0_P = (L0_P > 0).float() / (L0_P > 0).sum()
elif L0_P is not None and L0_MODE == "uniform":
    # Ignore the empirical distribution and sample level 0 uniformly.
    L0_P = torch.ones_like(L0_P) / L0_P.numel()
elif L0_P is not None and L0_MODE.startswith("smooth:"):
    # Mix in a small amount of uniform mass so rare codes are reachable.
    eps = float(L0_MODE.split(":", 1)[1])
    L0_P = (1.0 - eps) * L0_P + eps * torch.ones_like(L0_P) / L0_P.numel()

# Unconditional tasks (LM1B, OWT, TinyStories) define no data files for the
# split: they sample from the model instead of conditioning on an input.
UNCOND = not CFG.data or not CFG.data.get(split)
ADJ = [float(v) for v in adj.split(",")] if adj else None

if UNCOND:
    # Each shard owns complete seeds, not a fraction of a seed, so per-seed
    # metrics can still be computed after merging.
    seeds = [int(x) for x in CFG.eval.seeds][a.shard::a.nshards]
    n = a.n or int(CFG.eval.n_samples)
    items, refs, idxs, src_tok = [], [], [], None
else:
    # Conditional tasks: inputs and references are parallel pickles.
    items = pickle.load(open(os.path.join(dd, CFG.data[split]), "rb"))[:n]
    refs = pickle.load(open(os.path.join(dd, CFG.data[f"{split}_refs"]), "rb"))[:n]
    assert len(items) == len(refs), f"{len(items)} inputs vs {len(refs)} references"
    # Strided split: shard k takes positions k, k+nshards, k+2*nshards, ...
    idxs = list(range(a.shard, len(items), a.nshards))
    # Text tasks store raw strings, so an HF tokenizer is needed to build the
    # prefix. Sudoku stores already-tokenized id lists, hence src_tok=None.
    src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None


def prefix(item):
    """Return exactly ``P`` prefix ids: truncate long inputs, right-pad short ones.

    Padding is later masked by ``prefix_pad_token_id`` inside the generator.
    """
    ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
    return (ids[:P] + [PAD] * P)[:P]


def pick_tokens(logits):
    """Choose one token per position, optionally with count-based penalties.

    The fast path is plain argmax (deterministic decoding). The slow path
    matches the sampler used during training/evaluation of the released
    checkpoints: it tracks how often each token has been emitted, penalizes
    tokens above their allowance, optionally rewards tokens that are still
    below it, and penalizes recently repeated tokens.
    """
    if freq <= 0 and not ADJ:
        return logits.argmax(-1)
    B, T, V = logits.shape
    lg = logits.to(F64)
    allow = (ALLOW[:V] * T)[None, :] if ALLOW is not None else None
    al = ADJ[0] if ADJ else 0.0          # base adjacency penalty
    aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1   # window: how many previous tokens
    ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0      # decay of the penalty with distance
    cnt = torch.zeros(B, V, dtype=F64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0:
            # Push down tokens already emitted more often than allowed; with
            # `mu > 0`, pull up tokens still below their allowance (deficit).
            excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
            step -= freq * excess
            if mu > 0 and allow is not None:
                step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        # Penalize repeats of the last `aw` tokens, decaying with distance.
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=F64, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=F64, device=lg.device))
    return out


# Most tasks stop a sample at the first EOS; this can be disabled in the YAML.
CUT_AT_EOS = bool(G.get("cut_at_eos", True))


def dec(ids):
    """Decode one sequence of pre-tokenizer ids to text."""
    ids = list(ids)
    if CUT_AT_EOS and EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


# Raw (pre-tokenizer) token ids for every sample. Used by the FLM perplexity
# protocol, which needs the exact ids rather than the decoded string.
RAW = []


def run(x, l0=None):
    """Generate and decode one batch.

    ``x`` is the prefix (B x P ids). ``l0`` optionally supplies the coarsest
    level for unconditional sampling; when None the generator produces it
    itself (query/AR/predicted level 0).
    """
    with torch.no_grad():
        # Coarse-to-fine generation: one parallel pass per level. The result
        # concatenates the codes of every level, coarse first.
        g = net.generate(l0, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg, cfg_null_eos=EOS, use_kv_cache=a.kv_cache)
        # Turn the code hierarchy into latent vectors. zl[-1] is the latent at
        # the finest level (full output length).
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        # Only the finest level is decoded to output tokens.
        ids = pick_tokens(tk.vqvae.decoder(zl[-1])).tolist()
        RAW.extend(ids)
        return [dec(r) for r in ids]


hyps, sample_seed = [], []
if UNCOND:
    # Unconditional: draw the coarsest code from the level-0 unigram, then let
    # the generator fill in the finer levels.
    L0 = tk.vqvae.vq.levels[0]
    for seed in seeds:
        # Seeding per shard is deterministic, so shard k always generates the
        # same samples for its seeds regardless of the other shards.
        torch.manual_seed(seed)
        for s in range(0, n, bs):
            B = min(bs, n - s)
            l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).to(DEV) if L0_P is not None else None
            # The single EOS token is the usual start-of-sequence prefix.
            hyps += run(torch.full((B, 1), EOS, device=DEV), l0)
            sample_seed += [seed] * B
        print(f"  shard{a.shard} seed {seed}: {n} samples", flush=True)
    idxs = list(range(len(hyps)))
    # `toks` (pre-tokenizer ids) and `raw_ids` are only needed by the genPPL
    # metrics; they are ignored by the other metric implementations.
    toks = [tk.pre_tokenizer.encode(h) for h in hyps]
    out = {"idxs": idxs, "hyps": hyps, "refs": [""] * len(hyps), "seeds": sample_seed, "toks": toks, "raw_ids": RAW}
else:
    # Conditional: run the shard's inputs in order, in batches.
    for s in range(0, len(idxs), bs):
        batch = idxs[s:s + bs]
        hyps += run(torch.tensor([prefix(items[i]) for i in batch], device=DEV))
        # Only shard 0 prints progress, otherwise the logs would interleave.
        if a.shard == 0 and (s // bs) % 10 == 0:
            print(f"  shard0 {s}/{len(idxs)}", flush=True)
    # References may be raw strings (text tasks) or token id lists (Sudoku).
    out = {"idxs": idxs, "hyps": hyps, "refs": [r if isinstance(r, str) else dec(r) for r in (refs[i] for i in idxs)]}

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pickle.dump(out, open(a.out, "wb"))
print(f"shard {a.shard}: {len(hyps)} -> {a.out}")
