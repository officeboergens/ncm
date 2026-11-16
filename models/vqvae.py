"""Frozen multiscale VQ-VAE: the tokenizer half of an NCM checkpoint.

Pipeline:

    pre-token ids --Encoder--> latent sequence z --MultiscaleResidualVQ--> z_hat
                                                        |
                                                        +--> Decoder --> pre-token logits

The quantizer represents ``z`` with a hierarchy of code sequences (coarse to
fine). The generator (``models/next_concept.py``) learns to predict those code
sequences, and ``decode`` turns a code sequence back into text tokens.
"""
import torch.nn as nn
import torch.nn.functional as F
from models.vae import Encoder, Decoder
from utils.registry import vector_quantizers
import torch
import hydra
from omegaconf import OmegaConf
from utils.misc import count_params

class VQVAE(nn.Module):
    def __init__(
        self,
        in_vocab_size,          # vocabulary of the pre-tokenizer (e.g. 50304 for GPT-2)
        context_length,         # fixed sequence length the encoder/decoder operate on
        embed_dim,              # latent width; the generator projects from this
        vqvae_vocab_size,       # codebook size per hierarchy level
        n_layers,               # attention head count for the encoder/decoder blocks
        vector_quantizer_config,  # selects and configures the quantizer
        compression_factor=4,   # >1 shortens the sequence before quantization
        dropout=0.1,
        pre_quant_groupnorm=None,
        pre_quant_dropout=None,
        use_qk_norm=False,
        enc_layers=0,           # extra encoder transformer blocks (0 = embedding lookup only)
        enc_attn_window=0,      # local attention window in the extra encoder blocks
        enc_conv_layers=0,      # extra convolutional blocks before quantization
        enc_conv_kernel=3,
        dec_layers=1,           # decoder transformer depth
        pre_quant_norm_type='group',
        enc_positional=True,
        dec_positional=False,
        **kwargs
    ):
        super().__init__()
        _ = kwargs
        self.embed_dim = embed_dim
        self.vqvae_vocab_size = vqvae_vocab_size
        self.in_vocab_size = in_vocab_size
        self.context_length = context_length

        # enc_layers=0 / dec_layers=1 reproduce the original architecture: encoder = wte+wpe lookup, decoder = one Block.
        self.encoder = Encoder(
            in_vocab_size, context_length, embed_dim, n_layers, compression_factor=compression_factor,
            dropout=dropout, enc_layers=enc_layers, use_qk_norm=use_qk_norm,
            enc_attn_window=enc_attn_window,
            enc_conv_layers=enc_conv_layers, enc_conv_kernel=enc_conv_kernel,
            use_pos_emb=enc_positional,
        )

        # Resolve the quantizer class by name from the registry (hydra).
        vq = hydra.utils.get_class(vector_quantizers[vector_quantizer_config.clss])
        self.vq = vq.from_config({ **{'vqvae_vocab_size': vqvae_vocab_size, 'embed_dim': embed_dim}, **dict(vector_quantizer_config)})

        self.decoder = Decoder(
            in_vocab_size, embed_dim, n_layers, compression_factor=compression_factor,
            dropout=dropout, use_qk_norm=use_qk_norm, dec_layers=dec_layers,
            use_pos_emb=dec_positional, context_length=context_length,
        )

        # affine=False: ||z-e||^2 is not scale-invariant, so a learnable gamma would inflate without bound.
        if not pre_quant_groupnorm:
            self.pre_quant_norm = None
        elif str(pre_quant_norm_type) == 'layer':
            self.pre_quant_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self._pre_quant_is_layer = True
        else:
            self.pre_quant_norm = nn.GroupNorm(int(pre_quant_groupnorm), embed_dim, affine=False)
            self._pre_quant_is_layer = False

        self.pre_quant_drop = (
            nn.Dropout(float(pre_quant_dropout)) if pre_quant_dropout else None
        )

        self.context_length = context_length

        total_params, non_embd_params = count_params(self, exclude_params=['wte', 'wpe'])

        print("VQVAE, Total parameter count: %.2fM" % (total_params/1e6,))
        print("VQVAE, Non embedding parameter count: %.2fM" % (non_embd_params/1e6,))

    def _encode_pre_quant(self, x):
        """Encoder output, optionally normalized/dropped out before quantization.

        Normalizing here keeps the latent scale stable, which matters for a
        nearest-neighbour codebook lookup.
        """
        z = self.encoder(x)
        if self.pre_quant_norm is not None:
            if getattr(self, '_pre_quant_is_layer', False):
                z = self.pre_quant_norm(z)
            else:
                # GroupNorm expects channels first.
                z = self.pre_quant_norm(z.transpose(1, 2)).transpose(1, 2)
        if self.pre_quant_drop is not None:
            z = self.pre_quant_drop(z)
        return z

    def forward(self, x):
        """Training path: encode, quantize, decode; also returns VQ losses/utilization."""
        z = self._encode_pre_quant(x)
        z_q, vq_loss, per_level_codebook_util, global_codebook_util, ppl = self.vq(z)
        x = self.decoder(z_q)
        return x, vq_loss, per_level_codebook_util, global_codebook_util, ppl

    @torch.no_grad()
    def encode(self, x):
        """Pre-token ids -> flattened concatenation of all hierarchy codes."""
        z = self._encode_pre_quant(x)
        idx = self.vq.z_to_idx(z)
        return idx

    @torch.no_grad()
    def encode_z(self, x):
        """Pre-token ids -> continuous latent (no quantization).

        Used as the encoder for a raw-token prefix in the generator.
        """
        return self._encode_pre_quant(x)

    @torch.no_grad()
    def decode(self, idx):
        """Codes -> pre-token logits."""
        z_q = self.vq.idx_to_z_q(idx)
        x = self.decoder(z_q)
        return x

    @torch.no_grad()
    def reconstruct(self, x):
        """Full autoencoder pass (no argmax), for reconstruction experiments."""
        z = self._encode_pre_quant(x)
        z_q, _, _ = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon

    @torch.no_grad()
    def get_ncm_inputs(self, pre_tokenized_ids: torch.Tensor):
        """Build the inputs of the next-concept generator from raw text ids.

        Returns the per-level ``z_hat`` inputs (coarse levels already upsampled
        to the full length and concatenated) and the target codes for levels
        after level 0.
        """
        idx = self.encode(pre_tokenized_ids)
        x = self.vq.idx_to_ncm_input(idx)
        return x, idx[:, self.vq.levels[0]:]

    @torch.no_grad()
    def reconstruct_multiscales(self, idx_pre_tokenized: torch.Tensor):
        """Reconstruct the input at every hierarchy level (coarse -> fine)."""
        B, _ = idx_pre_tokenized.shape
        z = self._encode_pre_quant(idx_pre_tokenized)
        _, multiscale_zqs = self.vq.z_to_idx(z, return_zlevels=True)
        reconstructions = []
        for z_q in multiscale_zqs:
            logits = self.decoder(z_q)
            x = torch.argmax(logits, dim=-1)
            reconstructions.append(x)

        # Reorder from "per level" to "per sample".
        reconstructions_reordered = [[] for _ in range(B)]
        for i in range(len(reconstructions)):
            for j in range(B):
                reconstructions_reordered[j].append(reconstructions[i][j].tolist())
        return reconstructions_reordered


if __name__ == "__main__":

    # Small smoke test with a manually specified level schedule.
    vqvae = VQVAE(
        in_vocab_size=1000,
        context_length=512,
        embed_dim=256,
        vqvae_vocab_size=512,
        n_layers=8,
        vector_quantizer_config=OmegaConf.create({
            'clss': 'multiscale_residual_vector_quantizer',
            'decay': 0.99,
            'commitment_cost': 0.25,
            'levels': [1, 2, 4, 8, 12, 16, 24, 32, 48, 56, 64, 72, 84, 96, 114, 128]
        })
    )

    print(vqvae)
    sample_input = torch.randint(0, 1000, (2, 512))
    output, vq_loss, _, _, ppl = vqvae(sample_input)

    print("Output shape:", output.shape)
    print("VQ Loss:", vq_loss.item())
