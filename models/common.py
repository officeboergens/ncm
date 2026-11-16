"""Transformer building blocks shared by the generator and the VQ-VAE.

Contains RoPE helpers, normalization layers, a GPT-style attention block with
optional grouped-query attention (GQA), QK norm, KV-cache support and optional
local/flex attention, plus several MLP and residual-scaling variants that are
enabled by flags in the model config.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# PyTorch's fused flex-attention is optional; configs that request it need a
# build where this import succeeds.
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    HAS_FLEX_ATTENTION = True
except ImportError:
    _flex_attention = None
    HAS_FLEX_ATTENTION = False


def precompute_rope_cos_sin(positions, head_dim, base=10000):
    """Precompute rotary embeddings for the given integer positions.

    Returns cos/sin tensors of shape (len(positions), head_dim). The frequencies
    are duplicated across the two halves so that ``apply_rotary_emb`` can treat
    the head vector as (x1, x2) pairs.
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    half = head_dim // 2
    # theta_i = base^(-2i/d), expressed with half the exponent because the
    # cos/sin tables are duplicated afterwards.
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """Rotate (..., head_dim) vectors by the supplied cos/sin tables."""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Rotate by 90 degrees: (x1, x2) -> (-x2, x1); then blend by angle.
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (bias is off in most configs)."""
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)


class RMSNorm(nn.Module):
    """Root-mean-square norm, applied per head for QK norm.

    Computed in fp32 and cast back, matching the Llama-style implementation.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x_f * rms).to(orig_dtype) * self.weight

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # GQA: K/V use n_kv_heads <= n_head; the default reproduces full MHA.
        self.n_kv_head = int(getattr(config, 'n_kv_heads', config.n_head))
        assert config.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_heads"
        _hd = config.n_embd // config.n_head
        self._kv_dim = self.n_kv_head * _hd
        # One fused projection produces queries plus (possibly fewer) keys/values.
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self._kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        assert hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # QK norm: RMSNorm over head_dim of Q and K, applied before RoPE as in Llama 3.
        self.use_qk_norm = bool(getattr(config, 'use_qk_norm', False))
        if self.use_qk_norm:
            head_dim = config.n_embd // config.n_head
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_mask=None, rope_cos_sin=None, block_mask=None, past_kv=None, return_kv=False):
        """past_kv=(k, v) from an earlier call is attended over before x; return_kv=True also returns this
        call's rotated (k, v). Nothing is stored on the module.

        Keeping the cache in arguments instead of module state makes the same
        block usable by both the fused eager path and a torch.compile'd path.
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split([self.n_embd, self._kv_dim, self._kv_dim], dim=2)
        hd = C // self.n_head
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, hd).transpose(1, 2)
        if self.n_kv_head != self.n_head:
            # GQA: repeat each K/V head so it lines up with the query heads.
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # past_kv already holds rotated K, so rotate only [past_len, past_len+T).
        if rope_cos_sin is not None:
            cos_full, sin_full = rope_cos_sin
            past_len = past_kv[0].size(2) if past_kv is not None else 0
            cos = cos_full[past_len:past_len + T]
            sin = sin_full[past_len:past_len + T]
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # k (without RoPE) and v are views into the fused qkv output; copy them so a kept cache doesn't hold it.
        new_kv = (k.contiguous(), v.contiguous()) if return_kv else None
        if past_kv is not None:
            # A cached call attends to the previous K/V followed by the new ones.
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        if block_mask is not None:
            # flex_attention is not autocast-aware, so align q/k to v's dtype by hand.
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
                k = k.to(v.dtype)
            y = _flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Causal only for multiple tokens with no explicit mask; a cached single token sees all.
            use_causal = True if attn_mask is None and T > 1 else False
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=use_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return (y, new_kv) if return_kv else y

class MLP(nn.Module):
    """Feed-forward block; the variant is chosen by config flags.

    - default: Linear -> GELU -> Linear with a 4x (or `mlp_ratio`) hidden size;
    - SwiGLU/GEGLU: gated two-projection variant, hidden size ~8/3*n_embd;
    - relu2: ReLU squared activation, as used in some diffusion transformers.
    """

    def __init__(self, config):
        super().__init__()
        self.use_swiglu = bool(getattr(config, "use_swiglu", False))
        self.use_geglu  = bool(getattr(config, "use_geglu", False))
        self.use_relu2  = bool(getattr(config, "use_relu2", False))
        self.gated_ffn  = self.use_swiglu or self.use_geglu
        if self.gated_ffn:
            # Rounded to a multiple of 8 for tensor-core friendliness.
            h = int(round(8 * config.n_embd / 3 / 8) * 8)
            self.w_gate = nn.Linear(config.n_embd, h, bias=config.bias)
            self.w_up   = nn.Linear(config.n_embd, h, bias=config.bias)
            self.c_proj = nn.Linear(h, config.n_embd, bias=config.bias)
            self.act    = nn.SiLU() if self.use_swiglu else nn.GELU()
        else:
            _ratio = float(getattr(config, 'mlp_ratio', 4.0))
            _h = int(round(_ratio * config.n_embd))
            self.c_fc    = nn.Linear(config.n_embd, _h, bias=config.bias)
            self.gelu    = nn.GELU()
            self.c_proj  = nn.Linear(_h, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.gated_ffn:
            x = self.c_proj(self.act(self.w_gate(x)) * self.w_up(x))
        elif self.use_relu2:
            x = self.c_proj(torch.relu(self.c_fc(x)) ** 2)
        else:
            x = self.c_proj(self.gelu(self.c_fc(x)))
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Pre-norm transformer block with optional parallel attention/MLP and scaling.

    Variants:
    - ``parallel_block``: attention and MLP both read the same normalized input
      and are added to the residual stream together (GPT-J style);
    - ``normformer``: extra LayerNorms on the attention/MLP outputs;
    - ``use_layerscale`` / ``use_rezero``: learnable per-channel/per-branch
      scales on the residual branches.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.parallel_block = bool(getattr(config, "parallel_block", False))
        self.normformer = bool(getattr(config, "normformer", False))
        if self.normformer:
            self.ln_post_attn = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_post_mlp = LayerNorm(config.n_embd, bias=config.bias)
        self.use_layerscale = bool(getattr(config, "use_layerscale", False))
        if self.use_layerscale:
            _ls = float(getattr(config, "layerscale_init", 1e-4))
            self.ls1 = nn.Parameter(_ls * torch.ones(config.n_embd))
            self.ls2 = nn.Parameter(_ls * torch.ones(config.n_embd))
        self.use_rezero = bool(getattr(config, "use_rezero", False))
        if self.use_rezero:
            self.rz1 = nn.Parameter(torch.zeros(1))
            self.rz2 = nn.Parameter(torch.zeros(1))

    def _ls_scale(self, h, which):
        """Apply the residual-branch scale for branch 1 (attn) or 2 (MLP)."""
        if self.use_layerscale:
            return (self.ls1 if which == 1 else self.ls2) * h
        if self.use_rezero:
            return (self.rz1 if which == 1 else self.rz2) * h
        return h

    def forward(self, x, attn_mask=None, rope_cos_sin=None, block_mask=None, past_kv=None, return_kv=False):
        kv = dict(past_kv=past_kv, return_kv=return_kv)
        if self.parallel_block:
            # Both branches use the same pre-norm; their outputs are summed once.
            n = self.ln_1(x)
            a = self.attn(n, attn_mask, rope_cos_sin=rope_cos_sin, block_mask=block_mask, **kv)
            if return_kv:
                a, new_kv = a
            m = self.mlp(n)
            if self.normformer:
                a = self.ln_post_attn(a); m = self.ln_post_mlp(m)
            x = x + self._ls_scale(a, 1) + self._ls_scale(m, 2)
            return (x, new_kv) if return_kv else x
        # Standard sequential block: attention then MLP, each pre-normed.
        a = self.attn(self.ln_1(x), attn_mask, rope_cos_sin=rope_cos_sin, block_mask=block_mask, **kv)
        if return_kv:
            a, new_kv = a
        if self.normformer:
            a = self.ln_post_attn(a)
        x = x + self._ls_scale(a, 1)
        m = self.mlp(self.ln_2(x))
        if self.normformer:
            m = self.ln_post_mlp(m)
        x = x + self._ls_scale(m, 2)
        return (x, new_kv) if return_kv else x

class Transformer(nn.Module):
    """Minimal base class: weight init and the interface subclasses implement."""
    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        _ = non_embedding
        raise NotImplementedError("This method should be implemented in subclasses.")

    def forward(self, x):
        _ = x
        raise NotImplementedError("This method should be implemented in subclasses.")

    def generate(self, idx):
        _ = idx
        raise NotImplementedError("This method should be implemented in subclasses.")
