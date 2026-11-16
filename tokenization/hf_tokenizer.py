"""Thin wrapper around a Hugging Face tokenizer.

Used for text tasks (XSum, WMT14) when the model's prefix is built from raw
text: the wrapper exposes the same small interface (encode/decode, special
token ids) as the other pre-tokenizers.
"""
from tokenization.tokenizer import Tokenizer
from transformers import AutoTokenizer


class HFTokenizer(Tokenizer):
    def __init__(self, model_id="gpt2", pad_token=None, eos_token=None, **kwargs):
        super().__init__()
        self.model_id = model_id
        # Any remaining kwargs (e.g. trust_remote_code) go to HF.
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

        if eos_token is not None:
            self.hf_tokenizer.eos_token = eos_token
        # Add the requested pad token only if it is genuinely new; otherwise
        # just select it (GPT-2, for example, has no pad token by default).
        if pad_token is not None:
            if self.hf_tokenizer.convert_tokens_to_ids(pad_token) in (
                None, self.hf_tokenizer.unk_token_id, self.hf_tokenizer.eos_token_id
            ) and pad_token != self.hf_tokenizer.eos_token:
                self.hf_tokenizer.add_special_tokens({"pad_token": pad_token})
            else:
                self.hf_tokenizer.pad_token = pad_token
        elif self.hf_tokenizer.pad_token is None:
            # Fall back to EOS so batched encoding has something to pad with.
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

        self.vocab_size = self.hf_tokenizer.vocab_size
        self.eos_token_id = self.hf_tokenizer.eos_token_id
        self.pad_token_id = self.hf_tokenizer.pad_token_id

    @classmethod
    def from_config(cls, config):
        # `special_tokens:` in the YAML becomes keyword arguments (pad/eos).
        special = dict(config.special_tokens) if getattr(config, 'special_tokens', None) else {}
        return cls(config.model_id, **special)

    def encode(self, text: str):
        return self.hf_tokenizer.encode(text)

    def decode(self, ids, skip_special_tokens: bool = False):
        return self.hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
