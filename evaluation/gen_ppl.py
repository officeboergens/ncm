"""Generative perplexity of generated text under a reference language model.

Used by ``scripts/eval_merge.py`` for the XSum (GPT-2) and unconditional
(GPT-2 large) metrics. The generated text is concatenated with its prompt and
scored with a causal LM; ``score_only_generated`` excludes the prompt tokens
from the loss.

The result is a corpus-level perplexity: total NLL over all scored tokens
divided by the number of scored tokens, exponentiated. Per-example values are
also returned.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenPPLResult:
    nll_sum: torch.Tensor      # total negative log-likelihood per example
    token_count: torch.Tensor  # scored tokens per example
    ppl: torch.Tensor          # per-example perplexity (exp of mean NLL)
    mean_nll: float            # corpus-level mean NLL per token
    mean_ppl: float            # corpus-level perplexity


@torch.no_grad()
def generation_perplexity(
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generated_texts: List[str],
    *,
    device: Optional[str] = None,
    score_only_generated: bool = True,
    batch_size: int = 8,
    max_length: Optional[int] = None,
) -> GenPPLResult:

    assert len(prompts) == len(generated_texts), "prompts and generated_texts must align"
    ref_model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ref_model.to(device)

    # Some tokenizers have no pad token; EOS is a safe substitute for batching.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Fast tokenizers expose character offsets, which let us find the first
    # generated token exactly (used to exclude the prompt from the loss).
    use_offsets = getattr(tokenizer, "is_fast", False)

    n = len(prompts)
    nll_sums = []
    tok_counts = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_prompts = prompts[batch_start:batch_end]
        batch_gens = generated_texts[batch_start:batch_end]

        # Score the continuation in context: full = prompt + generation.
        full_texts = [p + g for p, g in zip(batch_prompts, batch_gens)]

        _full_max = max_length if max_length is not None else 1024
        enc_full = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_full_max,
            return_offsets_mapping=use_offsets,
        ).to(device)

        input_ids = enc_full["input_ids"]
        attn_mask = enc_full["attention_mask"]
        offset_mapping = enc_full.get("offset_mapping", None)

        if offset_mapping is not None:
            offset_mapping = offset_mapping.cpu()
            attn_mask_cpu  = attn_mask.cpu()

        logits = ref_model(input_ids=input_ids, attention_mask=attn_mask).logits

        # Next-token prediction: logits at position t score input_ids[t+1].
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask   = attn_mask[:, 1:]

        # By default score every non-padding token, including the prompt.
        include = shift_mask.bool().clone()

        if score_only_generated:
            B = input_ids.size(0)
            T = input_ids.size(1)

            if offset_mapping is not None:
                # A real token is an attended position whose character span is
                # not the empty span used for padding/special tokens.
                real_token = (
                    (attn_mask_cpu == 1) &
                    ~((offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] == 0))
                )

                # Tokens whose start offset falls inside the prompt are prompt
                # tokens; the rest belong to the generation.
                prompt_char_lens = torch.tensor(
                    [len(p) for p in batch_prompts]
                ).unsqueeze(1)
                is_gen = real_token & (offset_mapping[:, :, 0] >= prompt_char_lens)

                # argmax returns 0 for all-False rows, so fix those up explicitly.
                no_gen    = ~is_gen.any(dim=1)
                first_gen = is_gen.long().argmax(dim=1)
                first_gen[no_gen] = T

                # In shift_labels (= input_ids[:, 1:]) the first generated token sits at first_gen - 1.
                start_pos_vec = (first_gen - 1).clamp(min=0).to(device)
                positions = torch.arange(T - 1, device=device).unsqueeze(0)
                include = include & (positions >= start_pos_vec.unsqueeze(1))
            else:
                # Slow path without offsets: tokenize the prompt separately and
                # use its length to find where the generation starts.
                enc_prompt = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                prompt_lens = enc_prompt["attention_mask"].sum(dim=1).to(device)
                # argmax on attn_mask gives the index of the first real token (N for left-padding with N pads).
                padding_offset = attn_mask.long().argmax(dim=1)
                for i in range(B):
                    start_pos = max(int((padding_offset[i] + prompt_lens[i]).item()) - 1, 0)
                    include[i, :start_pos] = False

        # Negative log-likelihood of the actual next token under the model.
        log_probs = F.log_softmax(shift_logits, dim=-1)
        nll_tok   = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        nll_sum   = (nll_tok * include.float()).sum(dim=1)
        tok_count = include.sum(dim=1).to(torch.long)

        nll_sums.append(nll_sum.detach().cpu())
        tok_counts.append(tok_count.detach().cpu())

    nll_sum   = torch.cat(nll_sums,   dim=0)
    tok_count = torch.cat(tok_counts, dim=0)

    # Per-example perplexity (guard against zero scored tokens).
    tok_count_safe  = torch.clamp(tok_count, min=1)
    mean_nll_per_ex = nll_sum / tok_count_safe.float()
    ppl_per_ex      = torch.exp(mean_nll_per_ex)

    # Corpus-level perplexity: one number over the whole dataset.
    total_nll = float(nll_sum.sum().item())
    total_tok = int(tok_count.sum().item())
    mean_nll  = total_nll / max(total_tok, 1)
    mean_ppl  = math.exp(mean_nll)

    return GenPPLResult(
        nll_sum=nll_sum,
        token_count=tok_count,
        ppl=ppl_per_ex,
        mean_nll=mean_nll,
        mean_ppl=mean_ppl,
    )
