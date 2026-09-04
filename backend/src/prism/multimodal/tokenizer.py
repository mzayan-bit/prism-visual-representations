"""Deterministic Tokenizer and Vocabulary for PRISM Multimodal Learning."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from prism.multimodal.contracts import TokenizedText
from prism.multimodal.enums import SpecialToken


class Vocabulary:
    """Deterministic vocabulary mapping tokens to integer IDs."""

    def __init__(self, lexical_tokens: list[str] | None = None) -> None:
        # Special tokens are pinned to indices 0..3
        self.special_tokens = [
            SpecialToken.PAD.value,
            SpecialToken.UNK.value,
            SpecialToken.BOS.value,
            SpecialToken.EOS.value,
        ]
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # Deterministic alphabetical ordering for unique lexical tokens
        unique_lexical = sorted(set(lexical_tokens or []))
        # Filter out any accidental special token duplication
        unique_lexical = [t for t in unique_lexical if t not in self.special_tokens]

        self.tokens = self.special_tokens + unique_lexical
        self.token_to_id: dict[str, int] = {
            token: idx for idx, token in enumerate(self.tokens)
        }
        self.id_to_token: dict[int, str] = dict(enumerate(self.tokens))

    @property
    def size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.tokens)

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of vocabulary tokens."""
        data = json.dumps(self.tokens, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def encode_token(self, token: str) -> int:
        """Map a single token string to token ID, returning UNK id if not found."""
        return self.token_to_id.get(token, self.unk_id)

    def decode_id(self, token_id: int) -> str:
        """Map a single token ID to token string, returning UNK if out of bounds."""
        return self.id_to_token.get(token_id, SpecialToken.UNK.value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize vocabulary to dictionary."""
        return {
            "tokens": list(self.tokens),
            "size": self.size,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        """Instantiate vocabulary from serialized dictionary."""
        lexical = [
            t for t in data["tokens"] if t not in [st.value for st in SpecialToken]
        ]
        vocab = cls(lexical)
        return vocab


class SimpleTokenizer:
    """Deterministic whitespace and punctuation tokenizer with fixed padding."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        max_length: int = 16,
        version: str = "v1.0",
    ) -> None:
        if max_length < 3:
            msg = (
                f"max_length must be at least 3 for BOS + 1 token + EOS, "
                f"got {max_length}"
            )
            raise ValueError(msg)
        self.vocab = vocabulary
        self.max_length = max_length
        self.version = version

    def normalize(self, text: str) -> str:
        """Lowercase and normalize punctuation."""
        text = text.lower().strip()
        # Separate punctuation from words with spaces
        text = re.sub(r"([.,!?;:\'\"()\[\]{}])", r" \1 ", text)
        # Collapse multiple whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Convert text string into list of clean lexical token strings."""
        norm_text = self.normalize(text)
        if not norm_text:
            return []
        # Filter out standalone punctuation or keep simple alphanumeric tokens
        raw_tokens = norm_text.split(" ")
        tokens = [t for t in raw_tokens if t and re.match(r"^[a-zA-Z0-9_\-]+$", t)]
        return tokens

    def encode(self, text: str) -> TokenizedText:
        """Tokenize, frame with BOS/EOS, truncate, and pad to max_length."""
        raw_tokens = self.tokenize(text)

        # Max lexical tokens allowed: max_length - 2 (for BOS and EOS)
        max_lexical = self.max_length - 2
        truncated_lexical = raw_tokens[:max_lexical]

        # Full token string sequence: [BOS, ...truncated_lexical, EOS, ...PADs]
        full_tokens = [
            SpecialToken.BOS.value,
            *truncated_lexical,
            SpecialToken.EOS.value,
        ]
        valid_len = len(full_tokens)

        # Token IDs
        token_ids = [self.vocab.encode_token(t) for t in full_tokens]

        # Padding
        pad_count = self.max_length - valid_len
        token_ids += [self.vocab.pad_id] * pad_count
        full_tokens += [SpecialToken.PAD.value] * pad_count

        # Attention mask (1 for real token including BOS/EOS, 0 for PAD)
        attention_mask = [1] * valid_len + [0] * pad_count

        return TokenizedText(
            original_text=text,
            token_strings=full_tokens,
            token_ids=token_ids,
            sequence_length=valid_len,
            attention_mask=attention_mask,
        )

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a text string."""
        tokens: list[str] = []
        special_values = {st.value for st in SpecialToken}
        for tid in token_ids:
            tok = self.vocab.decode_id(tid)
            if skip_special and tok in special_values:
                continue
            tokens.append(tok)
        return " ".join(tokens)
