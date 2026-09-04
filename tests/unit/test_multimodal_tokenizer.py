"""Unit tests for deterministic tokenizer and vocabulary."""

from __future__ import annotations

import pytest

from prism.multimodal.enums import SpecialToken
from prism.multimodal.tokenizer import SimpleTokenizer, Vocabulary


def test_vocabulary_deterministic_ordering() -> None:
    """Verify special tokens occupy IDs 0..3 and words are sorted."""
    words = ["zebra", "apple", "banana", "apple"]
    vocab = Vocabulary(words)

    # Pinned special tokens
    assert vocab.encode_token(SpecialToken.PAD.value) == 0
    assert vocab.encode_token(SpecialToken.UNK.value) == 1
    assert vocab.encode_token(SpecialToken.BOS.value) == 2
    assert vocab.encode_token(SpecialToken.EOS.value) == 3

    # Sorted lexical tokens
    assert vocab.decode_id(4) == "apple"
    assert vocab.decode_id(5) == "banana"
    assert vocab.decode_id(6) == "zebra"
    assert vocab.size == 7

    # Fingerprint stability
    fp1 = vocab.fingerprint
    vocab2 = Vocabulary(["zebra", "banana", "apple"])
    assert vocab2.fingerprint == fp1


def test_vocabulary_serialization() -> None:
    """Verify vocabulary to_dict and from_dict roundtrip."""
    vocab = Vocabulary(["red", "blue", "green", "circle"])
    data = vocab.to_dict()

    restored = Vocabulary.from_dict(data)
    assert restored.size == vocab.size
    assert restored.fingerprint == vocab.fingerprint
    assert restored.tokens == vocab.tokens


def test_tokenizer_encoding_and_padding() -> None:
    """Verify tokenizer lowercasing, BOS/EOS framing, padding, and attention mask."""
    vocab = Vocabulary(["a", "red", "square", "on", "the", "left"])
    tokenizer = SimpleTokenizer(vocab, max_length=8)

    # Encode standard text: "A RED Square on the left"
    encoded = tokenizer.encode("A RED Square on the left.")
    # Tokens: [<BOS>, a, red, square, on, the, <EOS>, <PAD>] (fits max_length 8)
    assert len(encoded.token_ids) == 8
    assert encoded.token_ids[0] == vocab.bos_id
    assert encoded.token_strings[0] == SpecialToken.BOS.value

    # Check that EOS is present
    assert SpecialToken.EOS.value in encoded.token_strings
    eos_idx = encoded.token_strings.index(SpecialToken.EOS.value)
    assert eos_idx < 8

    # Attention mask must be 1 for valid tokens and 0 for PAD
    for idx, tok in enumerate(encoded.token_strings):
        if tok == SpecialToken.PAD.value:
            assert encoded.attention_mask[idx] == 0
            assert encoded.token_ids[idx] == vocab.pad_id
        else:
            assert encoded.attention_mask[idx] == 1


def test_tokenizer_unknown_tokens() -> None:
    """Verify unknown words map to UNK ID."""
    vocab = Vocabulary(["circle", "blue"])
    tokenizer = SimpleTokenizer(vocab, max_length=6)

    encoded = tokenizer.encode("giant triangle")
    # "giant" and "triangle" should map to unk_id (1)
    tok_ids = encoded.token_ids
    assert tok_ids[1] == vocab.unk_id
    assert tok_ids[2] == vocab.unk_id


def test_tokenizer_minimum_length_validation() -> None:
    """Verify ValueError is raised if max_length < 3."""
    vocab = Vocabulary(["a"])
    with pytest.raises(ValueError, match="max_length must be at least 3"):
        SimpleTokenizer(vocab, max_length=2)
