"""Text -> dense vectors. One interface, a real offline implementation and a production one.

`HashingEncoder` is a genuine vector encoder — word-and-bigram feature hashing into a fixed
dimension, L2-normalized — so cosine similarity is meaningful and the whole engine runs with no
download and deterministically. It is a real baseline (the hashing trick), not a mock; it just
has no learned semantics. `SentenceTransformerEncoder` is the production path: a learned
bi-encoder whose cosine space actually captures meaning. Both return L2-normalized rows so a dot
product is cosine.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> Vectors:
        """(len(texts), dim) float32, each row L2-normalized. For passages."""
        ...

    def encode_queries(self, texts: list[str]) -> Vectors:
        """The same, for search queries — some models want a query instruction."""
        ...


class HashingEncoder:
    """Feature-hashing bag of unigrams+bigrams. Deterministic, dependency-light, real vectors."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def encode(self, texts: list[str]) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _tokens(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for feat in features:
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode_queries(self, texts: list[str]) -> Vectors:
        return self.encode(texts)


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEncoder:
    """Production encoder: a learned bi-encoder. Lazy, never run here.

    Default `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions, 512 tokens, MIT): the same
    size as the earlier default `all-MiniLM-L6-v2` and markedly higher on MTEB retrieval (51.68
    average on its model card). Its card recommends an instruction in front of short queries —
    never passages — for retrieval, which `encode_queries` adds. For another model pass its name
    and its own `query_prefix` (empty for models that want none). Re-measure on your corpus
    (`entsearch eval --beir ...`) before trusting a leaderboard average."""

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        query_prefix: str | None = None,
    ) -> None:
        self.name = model
        self._model_name = model
        self._model: Any = None
        self.dim = 384  # bge-small-en-v1.5; corrected from the model after load
        if query_prefix is None:
            query_prefix = BGE_QUERY_INSTRUCTION if "bge-" in model and "-en" in model else ""
        self.query_prefix = query_prefix

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: list[str]) -> Vectors:
        model = self._ensure()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> Vectors:
        return self.encode([self.query_prefix + t for t in texts])
