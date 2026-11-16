"""Next-concept generator: predicts the multiscale code hierarchy of a VQ-VAE.

The model conditions on a prefix (a prompt, a source sentence, or a single
start token) and then generates codes coarse to fine. Every level is produced
in ONE parallel pass over a shorter-or-equal sequence; the next level is
conditioned on the accumulated latent of all previous levels. This is the
"next-concept" alternative to token-by-token autoregressive decoding.

Key ideas in this file:
- the frozen VQ-VAE is stored as a 1-tuple so it stays out of the module tree
  and the generator state dict;
- attention is block-structured: prefix tokens see only the prefix, and each
  target level sees the prefix plus itself (``block_diagonal``) or the prefix
  plus itself and all coarser levels (``block_causal``);
- RoPE positions restart at 0 for every target level, matching the per-level
  sequence layout;
- generation has an eager path and an optional prefix-KV-cache path.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Transformer, Block, LayerNorm, precompute_rope_cos_sin
from models.vqvae import VQVAE
from utils.misc import count_params

class _TiedCodebookHead(nn.Module):
    """Output head tied to a VQ codebook: logits are negative squared distances.

    Instead of learning an independent linear classifier, this projects the
    hidden state into the code embedding space and scores every code by
    ``2 z.E - ||E||^2`` (the ||z||^2 term is constant across the softmax).
    """

    def __init__(self, n_embd, codebook, learn_scale=True, trainable=False, use_bias=False):
        super().__init__()
        V, d = codebook.shape
        self.proj = nn.Linear(n_embd, d, bias=False)
        self.bias = nn.Parameter(torch.zeros(V)) if use_bias else None
        self.trainable = bool(trainable)
        if self.trainable:
            self.E = nn.Parameter(codebook.detach().clone())
        else:
            self.register_buffer("E", codebook.detach().clone(), persistent=False)
        self.out_features = V
        # Distance form, the inverse of z_to_idx's argmin ||z - E_i||^2: ||z||^2 cancels in the softmax, leaving 2 z.E_i - ||E_i||^2.
        E_sq = (codebook.detach() ** 2).sum(-1)
        if not self.trainable:
            self.register_buffer("E_sq", E_sq, persistent=False)
        self.register_buffer("d_scale", E_sq.mean().clamp(min=1e-12), persistent=False)
        self.log_inv_T = nn.Parameter(torch.tensor(0.0)) if learn_scale \
                         else torch.tensor(0.0)

    def forward(self, x):
        z = self.proj(x)
        E_sq = (self.E ** 2).sum(-1) if self.trainable else self.E_sq
        neg_d2 = (2.0 * (z @ self.E.t()) - E_sq) / self.d_scale
        out = neg_d2 * torch.exp(self.log_inv_T)
        return out if self.bias is None else out + self.bias


class _SharedHeadView(nn.Module):
    def __init__(self, head, bias):
        super().__init__()
        self.head = head
        self.bias = bias

    def forward(self, x):
        out = F.linear(x, self.head.weight)
        return out if self.bias is None else out + self.bias


class _SharedHeads(nn.Module):
    # one projection shared by every level; with per_level_bias each level owns its bias
    """Container that mimics a per-level head list while sharing one projection."""
    def __init__(self, head, n, per_level_bias=False):
        super().__init__()
        self.head = head
        self.n = int(n)
        self.per_level_bias = bool(per_level_bias)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(head.out_features)) for _ in range(self.n)]) if per_level_bias else None
        self._views = [_SharedHeadView(head, self.biases[i] if per_level_bias else None) for i in range(self.n)]

    def __getitem__(self, i):
        return self._views[i] if self.per_level_bias else self.head

    def __len__(self):
        return self.n


class NextConceptTransformer(Transformer):
    """Coarse-to-fine transformer over the VQ-VAE code hierarchy.

    Shapes:
    - prefix: B x P embeddings coming from the (frozen) VQ-VAE encoder, the
      raw token embedding, or an optional text context encoder;
    - target: B x sum(levels[1:]) embeddings, one chunk per generated level;
    - output: one logits tensor per level, concatenated along the sequence.
    """

    def __init__(self,
                 vqvae_local,
                 config
                 ):
        super().__init__()
        n_embd = config.n_embd
        n_layer = config.n_layer
        dropout = config.dropout
        bias = config.bias
        vocab_size = config.vocab_size
        attn_pattern = config.attn_pattern if hasattr(config, 'attn_pattern') else 'block_diagonal'
        self.config = config

        # A tuple keeps the frozen VQ-VAE out of the module tree and the state dict.
        self.vqvae_local = (vqvae_local,)

        self.levels = self.vqvae_local[0].vq.levels
        self.L = sum(self.levels)  # total number of codes at all levels

        init_std = math.sqrt(1 / n_embd)

        # query_level0: level 0 has only one code, so it is predicted from
        # learned query tokens instead of being generated position by position.
        # _lvl_first is the first level that lives in the target sequence.
        self.query_level0 = bool(getattr(config, 'query_level0', False))
        self.l0_ar = bool(getattr(config, 'l0_autoregressive', False)) and self.query_level0
        self._lvl_first = 0 if self.query_level0 else 1
        T = self.L if self.query_level0 else self.L - self.levels[0]
        P = config.prefix_len
        # Read optional flags once: getattr on a missing DictConfig key raises and swallows an exception whose
        # traceback cycles back to the caller's frame, so doing it in generate() kept every call's tensors alive
        # until the cyclic GC ran.
        self.raw_token_prefix = bool(getattr(config, 'raw_token_prefix', False))
        self.contextual_prefix = bool(getattr(config, 'contextual_prefix', False))

        # Per-level RoPE: prefix at [0..P-1], each target chunk reset to [0..l_k-1].
        self.use_rope = bool(getattr(config, 'use_rope', False))
        self.rope_base = int(getattr(config, 'rope_base', 10000))

        if self.use_rope:
            head_dim = n_embd // config.n_head
            levels_target = self.levels[self._lvl_first:]

            # One table for training/forward (prefix + all target levels) ...
            pos_full = torch.cat([
                torch.arange(P, dtype=torch.long),
                *[torch.arange(l, dtype=torch.long) for l in levels_target],
            ])
            cos_full, sin_full = precompute_rope_cos_sin(pos_full, head_dim, self.rope_base)
            self.register_buffer('rope_cos', cos_full, persistent=False)
            self.register_buffer('rope_sin', sin_full, persistent=False)

            # ... and one small table per generated level for generation, where
            # positions restart at 0 inside each level's chunk.
            for k, l_k in enumerate(levels_target):
                pos_k = torch.cat([
                    torch.arange(P, dtype=torch.long),
                    torch.arange(l_k, dtype=torch.long),
                ])
                cos_k, sin_k = precompute_rope_cos_sin(pos_k, head_dim, self.rope_base)
                self.register_buffer(f'rope_cos_gen_{k}', cos_k, persistent=False)
                self.register_buffer(f'rope_sin_gen_{k}', sin_k, persistent=False)
        else:
            T_max = max(P, max(self.levels[self._lvl_first:]))
            self.wpe = nn.Embedding(T_max, n_embd)
            wp_prefix = torch.arange(P, dtype=torch.long).view(1, -1)
            wp_target = torch.cat(
                [torch.arange(l, dtype=torch.long) for l in self.levels[self._lvl_first:]]
            ).view(1, -1)
            self.register_buffer('wp_prefix', wp_prefix)
            self.register_buffer('wp_1L', wp_target)
            nn.init.trunc_normal_(self.wpe.weight, mean=0, std=init_std)

        self.transformer = nn.ModuleDict(dict())

        self.transformer.update(dict(
            # Projects VQ-VAE latents (embed_dim) into the generator width.
            wte = nn.Linear(self.vqvae_local[0].embed_dim, n_embd, bias=bias)
        ))

        ctx_cfg = getattr(config, 'ctx_encoder', None)
        self.use_ctx_encoder = bool(getattr(ctx_cfg, 'enabled', False)) if ctx_cfg is not None else False
        self.ctx_pretrained = str(getattr(ctx_cfg, 'pretrained', '') or '') if ctx_cfg is not None else ''
        if self.use_ctx_encoder and self.ctx_pretrained:
            from transformers import T5EncoderModel
            self.ctx_t5 = T5EncoderModel.from_pretrained(self.ctx_pretrained)
            self.ctx_t5_finetune = bool(getattr(ctx_cfg, 'finetune', False))
            if not self.ctx_t5_finetune:
                self.ctx_t5.eval()
                for p in self.ctx_t5.parameters():
                    p.requires_grad = False
            d_ctx = int(self.ctx_t5.config.d_model)
            self.ctx_proj = nn.Linear(d_ctx, n_embd, bias=bias)
            self.ctx_pad_id = int(getattr(ctx_cfg, 'pad_id', 0))
            print(f"[ctx_encoder] pretrained={self.ctx_pretrained} d_model={d_ctx}->{n_embd} "
                  f"frozen={not self.ctx_t5_finetune} "
                  f"params={sum(p.numel() for p in self.ctx_t5.parameters())/1e6:.2f}M",
                  flush=True)
        elif self.use_ctx_encoder:
            n_ctx_layer = int(getattr(ctx_cfg, 'n_layer', 4))
            ctx_vocab = int(getattr(ctx_cfg, 'vocab_size', self.vqvae_local[0].in_vocab_size))
            self.ctx_wte = nn.Embedding(ctx_vocab, n_embd)
            if bool(getattr(ctx_cfg, 'init_from_gpt2', False)):
                from transformers import GPT2Model
                with torch.no_grad():
                    g = GPT2Model.from_pretrained('gpt2').wte.weight.data.to(
                        self.ctx_wte.weight.dtype)
                    tgt = float(getattr(ctx_cfg, 'init_scale_norm', 20.15))
                    g = g * (tgt / g.norm(dim=1).mean())
                    n = min(g.shape[0], ctx_vocab)
                    self.ctx_wte.weight[:n].copy_(g[:n])
                    if ctx_vocab > n:
                        nn.init.trunc_normal_(self.ctx_wte.weight[n:],
                                              mean=0.0, std=init_std)
            elif bool(getattr(ctx_cfg, 'init_from_vqvae', False)):
                with torch.no_grad():
                    src = self.vqvae_local[0].encoder.wte.weight
                    proj = self.transformer.wte.weight
                    w = src.to(proj.dtype) @ proj.t()
                    if self.transformer.wte.bias is not None:
                        w = w + self.transformer.wte.bias
                    V = min(w.shape[0], ctx_vocab)
                    self.ctx_wte.weight[:V].copy_(w[:V])
            else:
                nn.init.trunc_normal_(self.ctx_wte.weight, mean=0.0, std=init_std)
            self.ctx_blocks = nn.ModuleList([Block(config) for _ in range(n_ctx_layer)])
            self.ctx_ln = LayerNorm(n_embd, bias=bias) if n_ctx_layer > 0 else nn.Identity()

        puc_cfg = getattr(config, 'post_upsample_conv', None)
        self.use_post_upsample_conv = bool(getattr(puc_cfg, 'enabled', False)) if puc_cfg is not None else False
        if self.use_post_upsample_conv:
            k = int(getattr(puc_cfg, 'kernel_size', 3))
            assert k % 2 == 1, "post_upsample_conv.kernel_size must be odd for same-length padding"
            vq_dim = self.vqvae_local[0].embed_dim
            self.post_upsample_convs = nn.ModuleList([
                nn.Conv1d(vq_dim, vq_dim, kernel_size=k, padding=k // 2)
                for _ in range(len(self.levels) - (0 if self.query_level0 else 1))
            ])
            for conv in self.post_upsample_convs:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

        if config.use_level_encoding:
            self.transformer.update(dict(
                wle = nn.Embedding(len(self.levels) - (0 if self.query_level0 else 1), n_embd)
            ))

        self.transformer.update(dict(
            drop = nn.Dropout(dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))

        _n_heads = len(self.levels) - (0 if self.query_level0 else 1)
        # One output head per generated level; vocab sizes may differ per level.
        assert len(vocab_size) == _n_heads, \
            f"vocab_size length {len(vocab_size)} must equal {_n_heads} " \
            f"(query_level0={self.query_level0}, levels={len(self.levels)})"
        self.vocab_sizes = list(vocab_size)

        if bool(getattr(config, 'shared_head_mlp', False)):
            raise ValueError("shared_head_mlp is not supported")
        self.shared_output_head = bool(getattr(config, 'shared_output_head', False))
        if self.shared_output_head:
            assert len(set(self.vocab_sizes)) == 1, "shared_output_head requires equal vocab sizes"
            _plb = bool(getattr(config, 'shared_head_per_level_bias', False))
            _hb = bool(getattr(config, 'untied_head_bias', True)) and not _plb
            self.lm_heads = _SharedHeads(nn.Linear(n_embd, self.vocab_sizes[0], bias=_hb),
                                         len(self.vocab_sizes), per_level_bias=_plb)
        elif bool(getattr(config, 'tie_heads_to_codebook', False)):
            _vq = self.vqvae_local[0].vq
            _ls = bool(getattr(config, 'tie_heads_learn_scale', True))
            _tr = bool(getattr(config, 'tie_heads_trainable', False))
            _bi = bool(getattr(config, 'tie_heads_bias', False))
            self.lm_heads = nn.ModuleList([
                _TiedCodebookHead(n_embd, _vq.embeddings[i + self._lvl_first].weight,
                                  learn_scale=_ls, trainable=_tr, use_bias=_bi)
                for i in range(len(self.vocab_sizes))
            ])
        else:
            _hb = bool(getattr(config, 'untied_head_bias', True))
            self.lm_heads = nn.ModuleList([
                nn.Linear(n_embd, vs, bias=_hb) for vs in self.vocab_sizes
            ])

        if self.query_level0:
            # Learned queries that stand in for the (single) level-0 position.
            self.l0_queries = nn.Parameter(
                torch.zeros(1, self.levels[0], self.vqvae_local[0].embed_dim))
            nn.init.trunc_normal_(self.l0_queries, mean=0.0, std=init_std)

        self.prefix_len = P

        _ppid = getattr(config, 'prefix_pad_token_id', None)
        self.prefix_pad_id = int(_ppid) if _ppid is not None else None

        self.predict_level0 = bool(getattr(config, 'predict_level0', False))
        self.max_vocab = max(self.vocab_sizes + [self.vqvae_local[0].vq.vqvae_vocab_sizes[0]])
        if self.predict_level0:
            v0 = self.vqvae_local[0].vq.vqvae_vocab_sizes[0]
            assert P >= self.levels[0], (
                f"predict_level0 reads the last {self.levels[0]} prefix positions; "
                f"prefix_len={P} is too short"
            )
            if bool(getattr(config, 'tie_heads_to_codebook', False)):
                self.lm_head_l0 = _TiedCodebookHead(
                    n_embd, self.vqvae_local[0].vq.embeddings[0].weight,
                    learn_scale=bool(getattr(config, 'tie_heads_learn_scale', True)),
                    trainable=bool(getattr(config, 'tie_heads_trainable', False)),
                    use_bias=bool(getattr(config, 'tie_heads_bias', False)))
            else:
                self.lm_head_l0 = nn.Linear(n_embd, v0, bias=False)

        assert attn_pattern in ['block_diagonal', 'block_causal'], "attn_pattern must be either 'block_diagonal' or 'block_causal'"

        d = torch.cat([torch.full((l,), i) for i, l in enumerate(self.levels[self._lvl_first:])]).view(1, T, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)

        if attn_pattern == 'block_diagonal':
            # Every target position may attend only to positions of its own level.
            target_mask = torch.where(d == dT, 0., -torch.inf).reshape(T, T)
        else:
            # 'block_causal': a level may also see all coarser levels.
            target_mask = torch.where(d >= dT, 0., -torch.inf).reshape(T, T)
        if self.l0_ar:
            # Level 0 (when generated) is autoregressive within its chunk.
            _l0 = self.levels[0]
            target_mask[:_l0, :_l0] = torch.triu(
                torch.full((_l0, _l0), -torch.inf), diagonal=1)

        # Full mask over [prefix ; target]. Prefix rows only see the prefix
        # (top-right is -inf); target rows see the prefix and the allowed
        # target positions.
        top_left    = torch.zeros(P, P)
        top_right   = torch.full((P, T), -torch.inf)
        bottom_left = torch.zeros(T, P)
        bottom_right = target_mask

        full_mask = torch.cat([
            torch.cat([top_left,    top_right],   dim=1),
            torch.cat([bottom_left, bottom_right], dim=1),
        ], dim=0).reshape(1, 1, P + T, P + T)

        self.register_buffer('attn_mask_for_masking', full_mask.contiguous())

        self.use_flex_attention = bool(getattr(config, 'use_flex_attention', False))
        lvl_full = torch.cat([
            torch.full((P,), -1, dtype=torch.long),
            torch.cat([torch.full((l,), i, dtype=torch.long)
                       for i, l in enumerate(self.levels[self._lvl_first:])]),
        ])
        self.register_buffer('flex_lvl_full', lvl_full, persistent=False)
        self._flex_block_masks = {}

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        total_params, non_embd_params = count_params(self, exclude_params=['wte', 'wle', 'wpe'])
        
        print("NCP Transformer, Total parameter count: %.2fM" % (total_params/1e6,))
        print("NCP Transformer, Non embedding parameter count: %.2fM" % (non_embd_params/1e6,))
        
    def _get_flex_block_mask(self, device):
        """Build (and cache) the flex-attention block mask for the full sequence."""
        key = str(device)
        if key not in self._flex_block_masks:
            from torch.nn.attention.flex_attention import create_block_mask
            P = self.prefix_len
            S = P + (self.L - self.levels[0])
            lvl = self.flex_lvl_full.to(device)
            block_causal = getattr(self.config, 'attn_pattern', 'block_diagonal') == 'block_causal'

            def mask_mod(b, h, q_idx, kv_idx):
                qc = torch.clamp(q_idx, max=S - 1)
                kc = torch.clamp(kv_idx, max=S - 1)
                q_pref = q_idx < P
                kv_pref = kv_idx < P
                if block_causal:
                    same = (lvl[kc] <= lvl[qc]) & (lvl[kc] >= 0)
                else:
                    same = (lvl[qc] == lvl[kc]) & (lvl[kc] >= 0)
                return torch.where(q_pref, kv_pref, kv_pref | same)

            block_size = int(getattr(self.config, 'flex_block_size', 128))
            self._flex_block_masks[key] = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device,
                BLOCK_SIZE=block_size,
            )
        return self._flex_block_masks[key]

    def _apply_post_upsample_convs_multi(self, x_target):
        refined_chunks = []
        offset = 0
        for i, level in enumerate(self.levels[self._lvl_first:]):
            chunk = x_target[:, offset:offset + level, :]
            delta = self.post_upsample_convs[i](
                chunk.transpose(-1, -2)
            ).transpose(-1, -2)
            refined_chunks.append(chunk + delta)
            offset += level
        return torch.cat(refined_chunks, dim=1)

    def _apply_post_upsample_conv_single(self, chunk, level_idx_in_target):
        delta = self.post_upsample_convs[level_idx_in_target](
            chunk.transpose(-1, -2)
        ).transpose(-1, -2)
        return chunk + delta

    def _encode_context(self, prefix_ids):
        """Encode a raw-token prefix with the optional context encoder.

        Two options: a pretrained (usually frozen) T5 encoder, or a small local
        stack trained with the generator. Returns B x P x n_embd.
        """
        if self.ctx_pretrained:
            am = (prefix_ids != self.ctx_pad_id).long()
            if self.ctx_t5_finetune:
                h = self.ctx_t5(input_ids=prefix_ids, attention_mask=am).last_hidden_state
            else:
                with torch.no_grad():
                    h = self.ctx_t5(input_ids=prefix_ids,
                                    attention_mask=am).last_hidden_state
            return self.ctx_proj(h.to(self.ctx_proj.weight.dtype))
        h = self.ctx_wte(prefix_ids)
        rope_p = (self.rope_cos[:self.prefix_len],
                  self.rope_sin[:self.prefix_len]) if self.use_rope else None
        P = self.prefix_len
        ctx_mask = torch.zeros(1, 1, P, P, device=h.device)
        for block in self.ctx_blocks:
            h = block(h, attn_mask=ctx_mask, rope_cos_sin=rope_p)
        return self.ctx_ln(h)

    def forward(self, x, prefix_ids=None, prefix_pad_mask=None):
        """Training path: prefix latents (first P positions) + target latents.

        ``x`` is the concatenation of the prefix latent and the per-level
        target latents, exactly what ``VQVAE.get_ncm_inputs`` returns. Returns
        concatenated logits for all generated levels.
        """
        if self.use_ctx_encoder:
            assert prefix_ids is not None, "ctx_encoder requires prefix_ids"
            prefix_emb = self._encode_context(prefix_ids)
            x_target = x
            if self.use_post_upsample_conv:
                x_target = self._apply_post_upsample_convs_multi(x_target)
            x = self.transformer.wte(x_target)
            if not self.use_rope:
                x = x + self.wpe(self.wp_1L)
            if self.config.use_level_encoding:
                x = x + self.transformer.wle(self.lvl_1L)
            return self._run_stack(prefix_emb, x, prefix_pad_mask)

        z_prefix = x[:, :self.prefix_len, :]
        x_target = x[:, self.prefix_len:, :]

        if self.use_post_upsample_conv:
            x_target = self._apply_post_upsample_convs_multi(x_target)

        prefix_emb = self.transformer.wte(z_prefix)
        x = self.transformer.wte(x_target)

        if not self.use_rope:
            prefix_emb = prefix_emb + self.wpe(self.wp_prefix)
            x = x + self.wpe(self.wp_1L)

        if self.config.use_level_encoding:
            x += self.transformer.wle(self.lvl_1L)

        return self._run_stack(prefix_emb, x, prefix_pad_mask)

    def build_l0_query_input(self, batch_size, dtype=None):
        q = self.l0_queries.expand(batch_size, -1, -1)
        return q if dtype is None else q.to(dtype)

    def build_l0_ar_input(self, l0_idx, dtype=None):
        B = l0_idx.shape[0]
        prev = self.vqvae_local[0].vq.embeddings[0](l0_idx[:, :self.levels[0] - 1])
        x = torch.cat([self.l0_queries[:, :1].expand(B, -1, -1).to(prev.dtype), prev], dim=1)
        return x if dtype is None else x.to(dtype)

    def _prefix_pad_bias(self, pad_mask, S, dtype, device):
        B, P = pad_mask.shape
        bias = torch.zeros(B, 1, 1, S, dtype=dtype, device=device)
        bias[:, :, :, :P] = torch.where(
            pad_mask[:, None, None, :],
            torch.tensor(float('-inf'), dtype=dtype, device=device),
            torch.tensor(0.0, dtype=dtype, device=device))
        return bias

    def _level0_slice(self, x, prefix_pad_mask=None):
        P, l0 = self.prefix_len, self.levels[0]
        if prefix_pad_mask is None:
            return x[:, P - l0:P, :]
        last = (~prefix_pad_mask).sum(dim=1) - 1
        last = last.clamp(min=l0 - 1)
        offs = torch.arange(l0 - 1, -1, -1, device=x.device)
        pos = (last[:, None] - offs[None, :]).clamp(min=0, max=P - 1)
        return x.gather(1, pos[:, :, None].expand(-1, -1, x.shape[-1]))

    def _run_stack(self, prefix_emb, x, prefix_pad_mask=None):
        """Run the prefix and target through the transformer and project logits."""
        x = torch.cat([prefix_emb, x], dim=1)

        attn_mask = self.attn_mask_for_masking
        if prefix_pad_mask is not None:
            attn_mask = attn_mask + self._prefix_pad_bias(
                prefix_pad_mask, x.shape[1], attn_mask.dtype, x.device)

        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        if self.use_flex_attention:
            block_mask = self._get_flex_block_mask(x.device)
            for block in self.transformer.blocks:
                x = block(x, rope_cos_sin=rope, block_mask=block_mask)
        else:
            for block in self.transformer.blocks:
                x = block(x, attn_mask=attn_mask, rope_cos_sin=rope)
        x = self.transformer.ln_f(x)

        x_target = x[:, self.prefix_len:, :]
        logits_list = []
        if self.predict_level0:
            l0_h = self._level0_slice(x, prefix_pad_mask)
            logits_list.append(self._pad_vocab(self.lm_head_l0(l0_h)))
        offset = 0
        for i, level in enumerate(self.levels[self._lvl_first:]):
            logits_list.append(self._pad_vocab(self.lm_heads[i](x_target[:, offset:offset + level, :])))
            offset += level
        return torch.cat(logits_list, dim=1)


    def _pad_vocab(self, logits):
        """Right-pad logits to the largest vocabulary with -inf.

        Different levels may have different codebook sizes; padding lets the
        per-level logits be concatenated/sampled uniformly.
        """
        pad = self.max_vocab - logits.shape[-1]
        if pad == 0:
            return logits
        return F.pad(logits, (0, pad), value=torch.finfo(logits.dtype).min)

    # The KV-cached generation path as two side-effect-free steps: encode the prefix once and keep each block's
    # K/V, then run one level's tokens over them. Nothing is stored on the modules, so they can also be wrapped
    # with torch.compile.
    def _kv_prefix(self, h, mask, rope):
        kv = []
        for block in self.transformer.blocks:
            h, k_v = block(h, attn_mask=mask, rope_cos_sin=rope, return_kv=True)
            kv.append(k_v)
        return h, kv

    def _kv_level(self, x, prefix_kv, mask, rope, hidx):
        for block, k_v in zip(self.transformer.blocks, prefix_kv):
            x = block(x, attn_mask=mask, rope_cos_sin=rope, past_kv=k_v)
        return self.lm_heads[hidx](self.transformer.ln_f(x))

    def _sample_level(self, logits, temperature, top_k, top_p):
        """CFG-aware wrapper around ``_sample_level_raw``.

        With classifier-free guidance the batch is doubled: rows [0:B] are the
        conditional logits and [B:2B] the null-prefix logits. The guidance
        weight grows with the level index (coarse levels are guided less).
        """
        _c = getattr(self, "_cfg", None)
        if _c is not None and _c["scale"] > 0.0:
            B = _c["B"]
            t = _c["scale"] * (_c["level"] / max(_c["nlev"] - 1, 1))
            logits = (1.0 + t) * logits[:B] - t * logits[B:]
            pred = self._sample_level_raw(logits, temperature, top_k, top_p)
            return torch.cat([pred, pred], dim=0)
        return self._sample_level_raw(logits, temperature, top_k, top_p)

    def _sample_level_raw(self, logits, temperature, top_k, top_p):
        """Temperature / top-k / top-p sampling; temperature 0 means argmax."""
        if temperature == 0:
            return logits.argmax(dim=-1)
        logits_flat = logits.reshape(-1, logits.shape[-1]) / temperature
        if top_k > 0:
            kth = torch.topk(logits_flat, top_k, dim=-1).values[:, -1, None]
            logits_flat = logits_flat.masked_fill(logits_flat < kth, -torch.inf)
        sorted_logits, sorted_idx = torch.sort(logits_flat, dim=-1, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits_flat = torch.zeros_like(logits_flat).scatter_(-1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(logits_flat, dim=-1), num_samples=1).view(logits.shape[:-1])

    def _decode_level_logits(self, x, prefix_emb, level_idx, pad_bias=None, hidx=None):
        P = self.prefix_len
        hidx = level_idx if hidx is None else hidx
        L_cur = self.levels[level_idx + 1]
        x = torch.cat([prefix_emb, x], dim=1)

        gen_mask = torch.zeros(1, 1, P + L_cur, P + L_cur, device=x.device)
        gen_mask[:, :, :P, P:] = -torch.inf
        if pad_bias is not None:
            gen_mask = gen_mask + F.pad(pad_bias, (0, L_cur))
        rope_k = (
            getattr(self, f'rope_cos_gen_{hidx}'),
            getattr(self, f'rope_sin_gen_{hidx}'),
        ) if self.use_rope else None
        for block in self.transformer.blocks:
            x = block(x, attn_mask=gen_mask, rope_cos_sin=rope_k)
        x = self.transformer.ln_f(x)
        return self.lm_heads[hidx](x[:, P:, :])

    @torch.no_grad()
    def generate(self, idx, prefix_tokens, temperature=1.0, top_k=50, top_p=0.9,
                 use_kv_cache=False, cfg_scale=0.0, cfg_null_eos=50256):
        """Generate the full code hierarchy for a batch of prefixes.

        Args:
            idx: optional level-0 codes (B x levels[0]). If None, level 0 is
                produced from the learned query tokens / level-0 head.
            prefix_tokens: B x P prefix ids (text tokens or VQ-VAE pre-tokens).
            temperature/top_k/top_p: sampling controls; temperature 0 = argmax.
            use_kv_cache: encode the prefix once and reuse its K/V at each
                level instead of re-running the prefix every time.
            cfg_scale: classifier-free guidance weight; 0 disables it.
            cfg_null_eos: token used as the first token of the null prefix.

        Returns:
            B x sum(levels) codes, ordered coarse to fine (level 0 first).
        """
        assert not (self.l0_ar and use_kv_cache), \
            "autoregressive level-0 generation supports the eager path only"

        dev = prefix_tokens.device

        self._cfg = None
        # CFG: double the batch to [document; null] before the prefix is encoded; cfg_scale=0 builds no null branch.
        if cfg_scale > 0.0:
            if self.prefix_pad_id is None:
                raise ValueError("cfg_scale needs prefix_pad_id to build the null prefix")
            _cB = prefix_tokens.shape[0]
            _null = torch.full_like(prefix_tokens, self.prefix_pad_id)
            _null[:, 0] = cfg_null_eos
            prefix_tokens = torch.cat([prefix_tokens, _null], dim=0)
            if idx is not None:
                idx = torch.cat([idx, idx], dim=0)
            self._cfg = {"scale": float(cfg_scale), "B": _cB,
                         "nlev": len(self.levels), "level": 0}

        B = prefix_tokens.shape[0]
        vq = self.vqvae_local[0].vq
        # Running reconstruction, updated level by level.
        z_hat = torch.zeros(B, self.levels[-1], self.vqvae_local[0].embed_dim, device=dev)

        # ---- Encode the prefix -------------------------------------------------
        if self.use_ctx_encoder:
            prefix_emb = self._encode_context(prefix_tokens[:, :self.prefix_len])
        else:
            if self.raw_token_prefix:
                # Prefix ids are already pre-tokens: either a plain embedding
                # lookup or a contextual encoder pass over them.
                _enc = self.vqvae_local[0].encoder
                if self.contextual_prefix and \
                        len(getattr(_enc, 'blocks', [])) > 0:
                    z_prefix = _enc.forward_nopos(prefix_tokens)[:, :self.prefix_len, :]
                else:
                    z_prefix = _enc.wte(prefix_tokens)[:, :self.prefix_len, :]
            else:
                # Text prefix: let the frozen VQ-VAE encoder produce latents.
                z_prefix = self.vqvae_local[0].encode_z(prefix_tokens)[:, :self.prefix_len, :]
            prefix_emb = self.transformer.wte(z_prefix)
        if not self.use_rope:
            prefix_emb = prefix_emb + self.wpe(self.wp_prefix)

        # Level and positional embeddings for the target chunks. RoPE is
        # applied inside attention; wpe/wle are used otherwise.
        if self.config.use_level_encoding:
            level_pos_embed = self.transformer.wle(self.lvl_1L)
        else:
            level_pos_embed = torch.zeros(1, self.L - self.levels[0], self.config.n_embd, device=dev)

        if not self.use_rope:
            level_pos_embed = level_pos_embed + self.wpe(self.wp_1L)

        P = self.prefix_len
        prefix_kv = None

        _pbias = None
        if self.prefix_pad_id is not None:
            # Masked (pad) prefix positions get -inf so they cannot be attended.
            _pm = (prefix_tokens[:, :P] == self.prefix_pad_id)
            _pbias = torch.zeros(_pm.shape[0], 1, 1, P, device=dev)
            _pbias[:, :, :, :P] = torch.where(
                _pm[:, None, None, :],
                torch.tensor(float('-inf'), device=dev),
                torch.tensor(0.0, device=dev))

        if idx is None and self.query_level0:
            # ---- Level 0 from learned queries (one parallel pass) ------------
            l0 = self.levels[0]
            _q0 = self.build_l0_query_input(B, prefix_emb.dtype)
            if self.use_post_upsample_conv:
                _q0 = self._apply_post_upsample_conv_single(_q0, 0)
            q = self.transformer.wte(_q0)
            if not self.use_rope:
                q = q + self.wpe(self.wp_1L[:, :l0])
            if self.config.use_level_encoding:
                q = q + self.transformer.wle(self.lvl_1L[:, :l0])
            h = torch.cat([prefix_emb, q], dim=1)
            m0 = torch.zeros(1, 1, P + l0, P + l0, device=dev)
            m0[:, :, :P, P:] = float('-inf')
            if self.l0_ar:
                m0[:, :, P:, P:] = torch.triu(
                    torch.full((l0, l0), float('-inf'), device=dev), diagonal=1)
            if _pbias is not None:
                m0 = m0 + F.pad(_pbias, (0, l0))
            rope_0 = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
            if self.l0_ar:
                _E0 = vq.embeddings[0]
                base_q = self.build_l0_query_input(B, prefix_emb.dtype)
                x_in, codes = base_q, []
                for t in range(l0):
                    q = self.transformer.wte(x_in)
                    if not self.use_rope:
                        q = q + self.wpe(self.wp_1L[:, :l0])
                    if self.config.use_level_encoding:
                        q = q + self.transformer.wle(self.lvl_1L[:, :l0])
                    h = torch.cat([prefix_emb, q], dim=1)
                    for block in self.transformer.blocks:
                        h = block(h, attn_mask=m0, rope_cos_sin=rope_0)
                    h = self.transformer.ln_f(h)
                    codes.append(self._sample_level(
                        self.lm_heads[0](h[:, P + t:P + t + 1, :]),
                        temperature, top_k, top_p))
                    if t + 1 < l0:
                        known = torch.cat(codes, dim=1)
                        x_in = torch.cat([base_q[:, :1], _E0(known).to(base_q.dtype),
                                          base_q[:, t + 2:]], dim=1)
                idx = torch.cat(codes, dim=1)
            elif use_kv_cache:
                # The prefix rows can't attend to the level-0 queries, so their K/V in this pass are the
                # prefix cache: keep them instead of encoding the prefix again.
                prefix_kv = []
                for block in self.transformer.blocks:
                    h, (k, v) = block(h, attn_mask=m0, rope_cos_sin=rope_0, return_kv=True)
                    prefix_kv.append((k[:, :, :P], v[:, :, :P]))
                h = self.transformer.ln_f(h)
                if self._cfg is not None: self._cfg["level"] = 0
                idx = self._sample_level(self.lm_heads[0](h[:, P:, :]), temperature, top_k, top_p)
            else:
                for block in self.transformer.blocks:
                    h = block(h, attn_mask=m0, rope_cos_sin=rope_0)
                h = self.transformer.ln_f(h)
                if self._cfg is not None: self._cfg["level"] = 0
                idx = self._sample_level(self.lm_heads[0](h[:, P:, :]), temperature, top_k, top_p)

        if idx is None:
            # ---- Level 0 from the prefix (predict_level0) --------------------
            # Reads the last levels[0] prefix positions and projects them.
            assert self.predict_level0, "generate(idx=None) requires model.predict_level0 or query_level0"
            rope_p = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
            h = prefix_emb
            pfx_mask = torch.zeros(1, 1, P, P, device=dev)
            if _pbias is not None:
                pfx_mask = pfx_mask + _pbias
            if use_kv_cache:  # this pass is exactly the prefix encoding the cache needs
                h, prefix_kv = self._kv_prefix(h, pfx_mask, rope_p)
            else:
                for block in self.transformer.blocks:
                    h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
            h = self.transformer.ln_f(h)
            l0_logits = self.lm_head_l0(self._level0_slice(h, _pm if self.prefix_pad_id is not None else None))
            if self._cfg is not None: self._cfg["level"] = 0
            idx = self._sample_level(l0_logits, temperature, top_k, top_p)

        generated_idx = [idx]
        idx_next = idx
        if use_kv_cache and prefix_kv is None:
            # Level 0 was given: encode the prefix once here; every level attends over its K/V.
            rope_p = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
            pfx_mask = torch.zeros(1, 1, P, P, device=prefix_emb.device)
            if _pbias is not None:
                pfx_mask = pfx_mask + _pbias
            _, prefix_kv = self._kv_prefix(prefix_emb, pfx_mask, rope_p)

        level_acum = self.levels[0] if self.query_level0 else 0
        # ---- Levels 1..K: one parallel pass per level ------------------------
        for level_idx in range(len(self.levels)-1):
            if self._cfg is not None:
                self._cfg["level"] = level_idx + 1

            # Embed the codes just sampled, add them to the running latent and
            # feed the reconstruction for this level to the transformer.
            z_q_level = vq.embeddings[level_idx](idx_next)
            z_hat, x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)

            if self.use_post_upsample_conv:
                x = self._apply_post_upsample_conv_single(x, level_idx + (1 - self._lvl_first))

            x = self.transformer.wte(x) + level_pos_embed[:, level_acum:level_acum + self.levels[level_idx+1], :]

            L_cur = self.levels[level_idx + 1]
            # Heads and RoPE buffers are indexed by absolute level when level 0 occupies slot 0.
            _hi = level_idx + (1 - self._lvl_first)
            rope_k = (
                getattr(self, f'rope_cos_gen_{_hi}'),
                getattr(self, f'rope_sin_gen_{_hi}'),
            ) if self.use_rope else None

            if use_kv_cache:
                # Target rows only: the prefix K/V is already cached.
                lvl_mask = torch.zeros(1, 1, L_cur, P + L_cur, device=x.device)
                if _pbias is not None:
                    lvl_mask = lvl_mask + F.pad(_pbias, (0, L_cur))
                logits = self._kv_level(x, prefix_kv, lvl_mask, rope_k, _hi)
            else:
                logits = self._decode_level_logits(x, prefix_emb, level_idx, _pbias, hidx=_hi)
            # All positions of this level are sampled in parallel.
            idx_next = self._sample_level(logits, temperature, top_k, top_p)

            generated_idx.append(idx_next)
            level_acum += self.levels[level_idx+1]

        out = torch.cat(generated_idx, dim=1)
        if self._cfg is not None:
            out = out[:self._cfg["B"]]
        self._cfg = None
        return out
    