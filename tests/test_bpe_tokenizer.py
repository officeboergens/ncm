"""Regression tests for BPETokenizer special-token id resolution.

The bug: ``special_token_ids`` was built by enumerating
``special_tokens.values()`` from 0, so ``eos_token_id``/``pad_token_id``
depended on config key order and only matched the vocabulary by accident.

Run with::

    python -m unittest discover -s tests
"""
import os
import unittest

from tokenization.bpe_tokenizer import BPETokenizer

EOS = "<|endoftext|>"
PAD = "<|pad|>"

# Optional integration test: populated by `pixi run download-checkpoints`.
TINYSTORIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "tinystories",
)


def make_tokenizer(vocab, special_tokens, merges=()):
    return BPETokenizer(vocab, list(merges), special_tokens)


class SpecialTokenIdResolutionTest(unittest.TestCase):
    def setUp(self):
        # Specials deliberately *not* at ids 0/1, which is where enumerate()
        # would have put them. Ids 0/1 are ordinary vocabulary entries.
        self.vocab = {
            0: b"a",
            1: b"b",
            7: EOS.encode("utf-8"),
            9: PAD.encode("utf-8"),
        }
        self.special_tokens = {"eos_token": EOS, "pad_token": PAD}

    def test_ids_come_from_vocab_not_enumeration(self):
        tok = make_tokenizer(self.vocab, self.special_tokens)
        self.assertEqual(tok.eos_token_id, 7)
        self.assertEqual(tok.pad_token_id, 9)
        self.assertEqual(tok.special_token_ids, {EOS: 7, PAD: 9})
        self.assertEqual(tok.special_tokens_reverse_ids, {7: EOS, 9: PAD})

    def test_ids_do_not_depend_on_config_order(self):
        eos_first = {"eos_token": EOS, "pad_token": PAD}
        pad_first = {"pad_token": PAD, "eos_token": EOS}
        a = make_tokenizer(self.vocab, eos_first)
        b = make_tokenizer(self.vocab, pad_first)
        self.assertEqual(
            (a.eos_token_id, a.pad_token_id), (b.eos_token_id, b.pad_token_id)
        )
        self.assertEqual((a.eos_token_id, a.pad_token_id), (7, 9))

    def test_encode_decode_roundtrip_for_special_tokens(self):
        tok = make_tokenizer(self.vocab, self.special_tokens)
        self.assertEqual(tok.encode(EOS), [7])
        self.assertEqual(tok.decode([7]), EOS)
        self.assertEqual(tok.decode([7, 9]), EOS + PAD)

    def test_skip_special_tokens_filters_real_special_ids(self):
        tok = make_tokenizer(self.vocab, self.special_tokens)
        # Id 0 is a normal token and must survive; 7 and 9 must be dropped.
        self.assertEqual(tok.decode([0, 7, 9], skip_special_tokens=True), "a")

    def test_released_vocab_layout_is_unchanged(self):
        # The released TinyStories vocab stores the specials at ids 0 and 1,
        # so the corrected lookup must still yield the legacy ids.
        vocab = {
            0: EOS.encode("utf-8"),
            1: PAD.encode("utf-8"),
            2: b"a",
        }
        tok = make_tokenizer(vocab, self.special_tokens)
        self.assertEqual((tok.eos_token_id, tok.pad_token_id), (0, 1))

    def test_missing_special_token_raises(self):
        with self.assertRaisesRegex(ValueError, "not in the BPE vocabulary"):
            make_tokenizer({0: b"a"}, {"eos_token": EOS})

    def test_no_special_tokens(self):
        tok = make_tokenizer({0: b"a"}, None)
        self.assertEqual(tok.special_token_ids, {})
        self.assertIsNone(tok.eos_token_id)
        self.assertIsNone(tok.pad_token_id)

    def test_only_eos_configured(self):
        tok = make_tokenizer(self.vocab, {"eos_token": EOS})
        self.assertEqual(tok.eos_token_id, 7)
        self.assertIsNone(tok.pad_token_id)


class PaddingUsesResolvedIdTest(unittest.TestCase):
    def test_vqvae_padding_uses_vocab_special_id(self):
        from types import SimpleNamespace

        from tokenization.vqvae_tokenizer import VQVAETokenizer

        vocab = {
            0: b"a",
            1: b"b",
            7: EOS.encode("utf-8"),
            9: PAD.encode("utf-8"),
        }
        pre = make_tokenizer(vocab, {"eos_token": EOS, "pad_token": PAD})
        # _adjust_length only needs `context_length`; a full VQ-VAE is not
        # required to verify which id is used for the tail padding.
        tk = VQVAETokenizer(pre, SimpleNamespace(context_length=4))
        self.assertEqual(
            tk._adjust_length([7, 8, 7, 8, 7]),
            [[7, 8, 7, 8], [7, 9, 9, 9]],
        )


VOCAB_PATH = os.path.join(TINYSTORIES_DIR, "bpe_vocab_8192.pickle")
MERGES_PATH = os.path.join(TINYSTORIES_DIR, "bpe_merges_8192.pickle")


class ReleasedTinyStoriesCheckpointTest(unittest.TestCase):
    @unittest.skipUnless(
        os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH),
        "TinyStories BPE files not downloaded (run `pixi run download-checkpoints`)",
    )
    def test_released_special_ids_match_legacy_values(self):
        from types import SimpleNamespace

        tok = BPETokenizer.from_config(
            SimpleNamespace(
                vocab_file_path=VOCAB_PATH,
                merges_file_path=MERGES_PATH,
                special_tokens={"eos_token": EOS, "pad_token": PAD},
            )
        )
        self.assertEqual((tok.eos_token_id, tok.pad_token_id), (0, 1))
        self.assertEqual(tok.special_token_ids, {EOS: 0, PAD: 1})
        self.assertEqual(tok.encode(EOS + PAD), [0, 1])
        self.assertEqual(
            tok.decode([0, 1], skip_special_tokens=True),
            "",
        )


if __name__ == "__main__":
    unittest.main()
