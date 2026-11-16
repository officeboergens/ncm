"""Merge the per-shard pickles from ``eval_shard.py`` and compute the metric.

``evaluate.py`` calls this after all shards have finished. The script is
metric-agnostic until the very end; the metric is selected by
``eval.metric`` in the config (bleu, rouge, exact, genppl, genppl_flm).

Each shard pickle stores the original dataset indices alongside the
hypotheses/references, so sorting by index reconstructs dataset order even
though shards finish at different times.
"""
import argparse, glob, os, pickle, sys
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--glob", required=True)
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
# Load all shards and flatten their contents. `rows` is sorted by dataset index
# so hypotheses and references line up independent of shard order.
parts = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(a.glob))]
assert parts, "no shard files"
rows = sorted((i, h, r) for p in parts for i, h, r in zip(p["idxs"], p["hyps"], p["refs"]))
# Fields only produced by unconditional runs. `.get` keeps the conditional
# tasks (which do not write these keys) working unchanged.
seeds = [s for p in parts for s in p.get("seeds", [])]
toks = [t for p in parts for t in p.get("toks", [])]
raw_ids = [r for p in parts for r in p.get("raw_ids", [])]
# Hypotheses in their original per-shard order. `seeds` was built in the same
# order, so these two lists can be grouped by seed below.
hyps_by_part = [h for p in parts for h in p["hyps"]]
hyps = [h for _, h, _ in rows]
refs = [r for _, _, r in rows]
print(f"\nmerged {len(parts)} shards -> {len(rows)} {CFG.task} outputs\n")


def bleu():
    """WMT14: cased/lowercased BLEU, chrF and output/reference length ratio."""
    import sacrebleu
    b = sacrebleu.corpus_bleu(hyps, [refs])
    bl = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
    c = sacrebleu.corpus_chrf(hyps, [refs])
    # Length ratio in whitespace-separated words: >1 means over-generation.
    lr = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
    print(f"{'BLEU':>8} {'BLEU(lc)':>9} {'chrF':>7} {'len ratio':>10}")
    print(f"{b.score:8.2f} {bl.score:9.2f} {c.score:7.2f} {lr:10.3f}")
    print(f"\n{b.format(signature=True) if hasattr(b, 'format') else ''}")


def rouge():
    """XSum: ROUGE-1/2/L plus generative perplexity under GPT-2.

    Imports are local so that a BLEU-only run does not need rouge_score,
    transformers or torch loaded.
    """
    import torch
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    from utils.misc import default_device

    dev = default_device()
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    g2 = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()

    # ROUGE is computed per example and then averaged (macro average).
    R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for h, r in zip(hyps, refs):
        v = sc.score(r, h)
        for k in R:
            R[k].append(v[k].fmeasure)
    # Empty prompts + score_only_generated => perplexity of the summary alone.
    with torch.no_grad():
        gp = generation_perplexity(lm, g2, [""] * len(hyps), hyps, device=dev,
                                   score_only_generated=True, batch_size=32).mean_ppl
    print(f"{'R-1':>7} {'R-2':>7} {'R-L':>7} {'genPPL':>8} {'words':>7}")
    print(f"{100 * np.mean(R['rouge1']):7.2f} {100 * np.mean(R['rouge2']):7.2f} {100 * np.mean(R['rougeL']):7.2f} "
          f"{gp:8.2f} {np.mean([len(h.split()) for h in hyps]):7.1f}")


def exact():
    """Sudoku: exact-grid accuracy, per-cell accuracy and legality of the grid.

    The generated/reference strings include '|' row separators; `cells` strips
    them to get the 81 cell characters.
    """
    def cells(t):
        return t.replace("|", "")

    def valid(t):
        # A grid is valid when it is 81 non-zero digits and every row, column
        # and 3x3 box is a permutation of 1..9 (i.e. a solved Sudoku).
        g = cells(t)
        if len(g) != 81 or not g.isdigit() or "0" in g:
            return False
        rows_ = [g[i * 9:(i + 1) * 9] for i in range(9)]
        cols_ = ["".join(r[j] for r in rows_) for j in range(9)]
        boxes = ["".join(rows_[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)) for br in (0, 3, 6) for bc in (0, 3, 6)]
        return all(sorted(x) == list("123456789") for x in rows_ + cols_ + boxes)

    n = len(hyps)
    exact_n = sum(h == r for h, r in zip(hyps, refs))
    # Cell accuracy averages character matches over the 81 cells per puzzle.
    cell = np.mean([np.mean([a == b for a, b in zip(cells(h), cells(r))]) for h, r in zip(hyps, refs)])
    valid_n = sum(valid(h) for h in hyps)
    print(f"{'solved %':>9} {'solved':>7} {'cell %':>7} {'valid %':>8}")
    print(f"{100 * exact_n / n:9.2f} {exact_n:4d}/{n} {100 * cell:7.2f} {100 * valid_n / n:8.2f}")


def genppl():
    """Unconditional text tasks: per-seed generative PPL + unigram entropy.

    Perplexity is measured with ``eval.ref_model`` (GPT-2 large) on the
    generated text; entropy is measured on the pre-tokenizer ids to describe
    the diversity of the samples. Reported mean/std are over seeds.
    """
    import math
    import torch
    from collections import Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    from utils.misc import default_device

    dev = default_device()
    ref = str(CFG.eval.ref_model)
    rtok = AutoTokenizer.from_pretrained(ref)
    rlm = AutoModelForCausalLM.from_pretrained(ref).to(dev).eval()

    def entropy(ids):
        # Shannon entropy (nats) of the empirical unigram distribution.
        c, m = Counter(ids), len(ids)
        return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0

    per = []
    # dict.fromkeys preserves seed order while de-duplicating.
    for seed in dict.fromkeys(seeds):
        # Select this seed's samples from the per-shard ordering.
        sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
        gen = [hyps_by_part[k] for k in sel]
        with torch.no_grad():
            ppl = generation_perplexity(rlm, rtok, [""] * len(gen), gen, device=dev, batch_size=32).mean_ppl
        ent = float(np.mean([entropy(toks[k]) for k in sel if toks[k]]))
        per.append((seed, ppl, ent))
    print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8}")
    for seed, ppl, ent in per:
        print(f"{seed:5d} {ppl:8.2f} {ent:8.4f}")
    g = np.array([p[1] for p in per]); e = np.array([p[2] for p in per])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {len(hyps_by_part)} samples)")


def genppl_flm():
    """LM1B under the FLM scoring protocol.

    Unlike `genppl`, this keeps special tokens in the decoded text, rescores
    only the first 128 reference-model tokens and computes a corpus-level
    perplexity (sum of NLLs / total tokens) rather than a mean of per-sample
    perplexities.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.misc import default_device

    dev = default_device()
    # Decode raw ids with the tokenizer the samples were generated with.
    gen_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
    ref = str(CFG.eval.ref_model)
    rtok = AutoTokenizer.from_pretrained(ref)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rlm = AutoModelForCausalLM.from_pretrained(ref).to(dev).eval()

    def corpus_ppl(texts):
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
        ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        nll_sum = tok_sum = 0.0
        with torch.no_grad():
            for i in range(0, ids.shape[0], 16):
                s_, m_ = ids[i:i + 16], attn[i:i + 16]
                logits = rlm(s_, attention_mask=m_)[0].transpose(-1, -2)
                nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                # Score up to and including the first EOS; ignore padding and
                # everything after the end of the sample.
                first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
                nll_sum += float((nll * valid).sum()); tok_sum += float(valid.sum())
        return float(np.exp(nll_sum / tok_sum))

    def entropy(ids):
        t = torch.tensor(ids)
        return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())

    per = []
    for seed in dict.fromkeys(seeds):
        sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
        # batch_decode keeps special tokens (the FLM protocol scores them).
        texts = gen_tok.batch_decode([raw_ids[k] for k in sel])
        per.append((seed, corpus_ppl(texts), float(np.mean([entropy(raw_ids[k]) for k in sel]))))
    print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8}")
    for seed, ppl, ent in per:
        print(f"{seed:5d} {ppl:8.2f} {ent:8.4f}")
    g = np.array([p[1] for p in per]); e = np.array([p[2] for p in per])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {len(raw_ids)} samples)")


# Metric dispatch: the config names the function to run.
{"bleu": bleu, "rouge": rouge, "exact": exact, "genppl": genppl, "genppl_flm": genppl_flm}[str(CFG.eval.metric)]()
