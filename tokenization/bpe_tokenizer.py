"""Byte-level BPE pre-tokenizer (GPT-2-style merges, own vocabulary).

The vocabulary and merge list are pickled at training time; ``encode`` applies
the GPT-2 pre-tokenization regex and then replays the merges in learned order,
``decode`` concatenates the raw bytes of each token.
"""
from tokenization.tokenizer import Tokenizer
import regex as re
import pickle


class BPETokenizer(Tokenizer):
    def __init__(self, vocab, merges, special_tokens=None):
        super().__init__()
        self.vocab = vocab                      # id -> token bytes
        self.reverse_vocab = {value: key for key, value in vocab.items()}  # token bytes -> id
        # Merge rank: lower means the pair was learned earlier and is applied first.
        self.merges = {merge: idx for idx, merge in enumerate(merges)}
        self.vocab_size = len(vocab)

        # special_tokens is a name -> token-string mapping, e.g.
        # {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}.
        self.special_tokens = special_tokens
        special = dict(special_tokens) if special_tokens else {}
        # Resolve each special token through the vocabulary. Enumerating the
        # values would only be correct if the vocab happened to place those
        # exact byte strings at ids 0, 1, ...; a newly trained vocab usually
        # appends them elsewhere (and the ids must not depend on config order).
        self.special_token_ids = {}
        for name, token in special.items():
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.reverse_vocab:
                raise ValueError(
                    f"Special token {name}={token!r} is not in the BPE vocabulary; "
                    f"it cannot be encoded or skipped reliably."
                )
            self.special_token_ids[token] = self.reverse_vocab[token_bytes]
        self.special_tokens_reverse_ids = {v: k for k, v in self.special_token_ids.items()}
        self.eos_token_id = self.special_token_ids.get(special.get("eos_token"))
        self.pad_token_id = self.special_token_ids.get(special.get("pad_token"))

    @classmethod
    def from_config(cls, config):
        # Paths in the config were rewritten to the checkpoint dir by eval_shard.py.
        with open(config.vocab_file_path, 'rb') as file:
            vocab = pickle.load(file)
        with open(config.merges_file_path, 'rb') as file:
            merges = pickle.load(file)
        return BPETokenizer(vocab, merges, config.special_tokens)

    def _get_byte_pairs(self, byt_str):
        """All adjacent pairs of a tuple of bytes (candidate merges)."""
        byte_pairs = []
        for idx in range(len(byt_str)-1):
            byte_pairs.append(tuple([byt_str[idx], byt_str[idx+1]]))
        return byte_pairs

    def encode(self, text: str):
        special_tokens = sorted(self.special_tokens.values(), reverse=True) if self.special_tokens != None else []

        # 1. Split the text around special tokens so they survive pre-tokenization.
        if special_tokens:
            pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
            text = re.split(f"({pattern})", text)
        else:
            text = [text]

        # 2. GPT-2 pre-tokenization: contractions, words, numbers, punctuation,
        #    and whitespace, keeping a leading space attached to the next word.
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        text = [re.findall(pattern, chunk) if (chunk and chunk not in special_tokens) else [chunk] for chunk in text]

        # 3. Represent every piece as a tuple of single bytes.
        parse_word = lambda word: tuple(bytes([byt])for byt in word)

        text = [[parse_word(word.encode("utf-8")) if not word in special_tokens else word for word in chunk] for chunk in text if chunk != ['']]

        new_words = []

        # 4. Replay the merges: repeatedly apply the lowest-ranked pair present
        #    in the current word until no learned pair remains.
        for chunk in text:
            for word in chunk:
                if word in special_tokens:
                    # Special tokens bypass merging and act as single ids.
                    new_words.append(word.encode("utf-8"))
                else:
                    merge_exists = True
                    new_word = word
                    while merge_exists:
                        earliest_merge_idx = len(self.merges)
                        earliest_merge = None
                        candidate_merges = self._get_byte_pairs(new_word)
                        for candidate_merge in candidate_merges:
                            if candidate_merge in self.merges and self.merges[candidate_merge] < earliest_merge_idx:
                                earliest_merge = candidate_merge
                                earliest_merge_idx = self.merges[candidate_merge]

                        if earliest_merge != None:
                            merge_exists = True
                            # Replace every non-overlapping occurrence left to right.
                            tmp = []
                            idx = 0
                            while idx < len(new_word):
                                if idx < len(new_word) - 1 and new_word[idx:idx+2] == earliest_merge:
                                    tmp.append(earliest_merge[0] + earliest_merge[1])
                                    idx += 2
                                else:
                                    tmp.append(new_word[idx])
                                    idx += 1
                            new_word = tuple(tmp)
                        else:
                            merge_exists = False
                            earliest_merge = None
                    # A fully merged word is a sequence of multi-byte tokens.
                    new_words += [w for w in new_word]
        return [self.reverse_vocab[byte_pair] for byte_pair in new_words]

    def decode(self, ids: list[int], skip_special_tokens: bool = False):

        if skip_special_tokens and self.special_tokens != None:
            ids = [id for id in ids if id not in self.special_token_ids.values()]

        # Concatenate the bytes of all tokens, then decode the result as UTF-8.
        byts_list = [self.vocab[id] for id in ids]
        tmp = []
        for byts in byts_list:
            tmp += list(byts)

        try:
            return bytes(tmp).decode("utf-8", errors="replace")
        except UnicodeDecodeError as e:
            # Truncated/invalid byte sequences decode to an empty string.
            _ = e
            return ''
