"""Sudoku pre-tokenizer.

Each board cell is one symbol, rows are separated by ``|``; the model also
sees a leading ``<BOS>`` token. The vocabulary is tiny (12 symbols), which is
why the eval config can decode the full 89-position solution directly.

Ids: 0 = blank (_), 1-9 = digits, 10 = row separator (|), 11 = BOS.
"""
from tokenization.tokenizer import Tokenizer


class SudokuTokenizer(Tokenizer):
    vocab_size = 12
    blank_token_id = 0
    row_separator_id = 10
    bos_token_id = 11
    # Sudoku has no padding/EOS: boards are fixed length (89 cells + separators).
    pad_token_id = None
    eos_token_id = None
    seq_len = 89          # 81 cells + 8 row separators
    prompt_len = 91       # BOS + full board; the generator's prefix length

    @classmethod
    def from_config(cls, config):
        # Nothing to configure: the mapping is fixed.
        _ = config
        return cls()

    def encode(self, text):
        # `<BOS>` is spelled out in the data as the single character "B".
        text = text.replace('<BOS>', 'B')
        chars = {'_': 0, '|': 10, 'B': 11, **{str(i): i for i in range(10)}}
        return [chars[c] for c in text if not c.isspace()]

    def decode(self, ids, skip_special_tokens=False):
        # Inverse mapping, used for both generated hypotheses and references.
        chars = ['_'] + list('123456789') + ['|', '<BOS>']
        return ''.join(chars[int(i)] for i in ids
                       if not (skip_special_tokens and int(i) == self.bos_token_id))
