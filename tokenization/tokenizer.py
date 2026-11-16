"""Abstract tokenizer interface shared by the concrete pre-tokenizers.

A pre-tokenizer maps text to a sequence of integer ids and back. It sits in
front of the VQ-VAE (see ``tokenization/vqvae_tokenizer.py``): the VQ-VAE sees
these ids as its input vocabulary and emits the multiscale code hierarchy used
by the generator.
"""
from collections.abc import Iterable
import itertools

class Tokenizer(object):
    def encode(self, text: str):
        """Text -> list of token ids. Implemented by subclasses."""
        _ = text
        raise NotImplementedError("This method should be implemented in subclasses")

    def encode_iterable(self, iterable: Iterable[str]):
        """Lazily concatenate the encodings of many strings into one iterator."""
        itr = iter([])
        for text in iterable:
            itr = itertools.chain(itr, self.encode(text))
        return itr

    @classmethod
    def from_config(cls, config):
        """Build a tokenizer from its section of the eval YAML. Implemented by subclasses."""
        _ = config
        raise NotImplementedError("This method should be implemented in subclasses")

    def encode_chunk(self, input_path, start, end):
        """Encode a byte range of a UTF-8 file (used to shard large corpora)."""
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            return self.encode(chunk)

    def decode(self, ids: list[int], skip_special_tokens: bool = False):
        """List of token ids -> text. Implemented by subclasses."""
        _ = ids
        raise NotImplementedError("This method should be implemented in subclasses")

    def decode_batch(self, ids_batch: list[list[int]], skip_special_tokens: bool = False):
        """Convenience wrapper: decode a batch one sequence at a time."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_batch]
