"""Cross-encoder re-ranking backend for :mod:`sparksage.retrieve`.

A *cross-encoder* re-scores each ``(query, document)`` pair with a single
Transformer pass that jointly attends over both texts (rather than encoding them
independently the way a bi-encoder / dense embedder does). That cross-attention
makes it the most accurate single-stage re-ranker family available -- the
classic lever, after chunking strategy, for closing the "the right block is in
the pool but not at the top" gap. In practice it is also far cheaper per query
than :class:`~sparksage.retrieve.LLMReranker` (no token-billed generation), so
it is the default recommendation once the candidate pool exceeds a handful of
chunks.

This backend implements the existing
:class:`~sparksage.retrieve.reranker.Reranker` protocol verbatim -- swap it in
wherever a :class:`~sparksage.retrieve.Reranker` is accepted (e.g. the
``reranker=`` slot of :class:`~sparksage.retrieve.Retriever`). No orchestrator
change is needed; that is the whole point of the protocol.

The ``sentence-transformers`` package is an *optional* dependency -- install it
with ``pip install 'sparksage[rerank]'``. It is imported lazily inside
``__init__`` (matching the convention used by every other optional SDK backend,
e.g. :class:`~sparksage.embed.backends.FaissVectorStore`), so importing this
module is always free; only constructing the reranker pulls the SDK.

Score convention: :class:`~sparksage.retrieve.reranker.LLMReranker` reports
relevance scores in ``(0, 1]``. A raw cross-encoder emits unbounded logits, so
by default we squash them through a numerically-stable sigmoid into ``(0, 1)``
(``apply_sigmoid=True``) -- keeping scores comparable in shape to the other
rerankers. Disable it (``apply_sigmoid=False``) to surface raw model logits.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from sparksage.retrieve.models import RetrievedChunk

#: Default cross-encoder model. A small, fast, English-trained MS-MARCO ranker;
#: pass ``model_name=`` for a multilingual / larger model (e.g.
#: ``"BAAI/bge-reranker-v2-m3"`` for CJK + multilingual, which pairs well with
#: SparkSage's dictionary-free CJK tokenizers).
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _stable_sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid (no overflow on large ``|x|``)."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _candidate_text(chunk: RetrievedChunk) -> str:
    """Render a candidate as the QA-aligned text the cross-encoder scores.

    Mirrors the fields :class:`~sparksage.retrieve.LLMReranker` renders and the
    schema's ``embedding_text`` composition (``name`` + question + answer), so
    the cross-encoder sees the same self-contained unit the rest of the
    pipeline does.
    """
    block = chunk.block
    return f"{block.name}\n{block.critical_question}\n{block.trusted_answer}"


class CrossEncoderReranker:
    """Re-rank candidate blocks with a ``sentence_transformers`` cross-encoder.

    Wraps :class:`sentence_transformers.CrossEncoder` behind the
    :class:`~sparksage.retrieve.Reranker` protocol. Each candidate is scored
    against the query in one joint cross-attention pass; candidates are then
    re-ordered best-first and (optionally) truncated to ``top_n``.

    The ``sentence-transformers`` package is an *optional* dependency -- install
    it with ``pip install 'sparksage[rerank]'``.

    Parameters
    ----------
    model_name:
        Hugging Face model id of a cross-encoder / re-ranker checkpoint. The
        default (``cross-encoder/ms-marco-MiniLM-L-6-v2``) is small, fast and
        English-trained; use ``"BAAI/bge-reranker-v2-m3"`` for a multilingual
        re-ranker (recommended when the corpus contains CJK).
    max_length:
        Forwarded to ``CrossEncoder`` as the per-pair max token length.
    device:
        Torch device (``"cpu"``, ``"cuda"``, ...) forwarded to ``CrossEncoder``.
    trust_remote_code:
        Forwarded to ``CrossEncoder`` (required by some community checkpoints).
    apply_sigmoid:
        When ``True`` (default) raw logits are squashed through a stable sigmoid
        into ``(0, 1)`` so scores match the ``(0, 1]`` shape the other rerankers
        report. Set ``False`` to surface raw logits.

    Examples
    --------
    >>> from sparksage.retrieve import Retriever                      # doctest: +SKIP
    >>> from sparksage.retrieve.backends import CrossEncoderReranker  # doctest: +SKIP
    >>> retriever = Retriever(                                        # doctest: +SKIP
    ...     registry, store, embedder,
    ...     reranker=CrossEncoderReranker(),
    ... )
    >>> retriever.search("how to deploy", k=5, use_rerank=True)       # doctest: +SKIP
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        max_length: int | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
        apply_sigmoid: bool = True,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "CrossEncoderReranker requires the 'sentence-transformers' "
                "package. Install it with: pip install 'sparksage[rerank]'"
            ) from exc
        if not isinstance(model_name, str) or not model_name.strip():
            raise TypeError("model_name must be a non-empty string")
        if not isinstance(apply_sigmoid, bool):
            raise TypeError("apply_sigmoid must be a bool")

        kwargs: dict[str, Any] = {}
        if max_length is not None:
            kwargs["max_length"] = max_length
        if device is not None:
            kwargs["device"] = device
        if trust_remote_code:
            kwargs["trust_remote_code"] = True

        self._model_name = model_name
        self._apply_sigmoid = apply_sigmoid
        self._model = CrossEncoder(model_name, **kwargs)

    @property
    def model_name(self) -> str:
        """The Hugging Face id of the wrapped cross-encoder checkpoint."""
        return self._model_name

    @property
    def apply_sigmoid(self) -> bool:
        """Whether raw logits are squashed into ``(0, 1)`` on output."""
        return self._apply_sigmoid

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-score ``chunks`` against ``query``; return best-first slice.

        Returns a new list of :class:`~sparksage.retrieve.RetrievedChunk` whose
        ``score`` is the (optionally sigmoid-normalized) cross-encoder relevance
        and whose ``rank`` is the 0-indexed position in the new ordering.
        """
        if top_n is not None:
            if not isinstance(top_n, int) or isinstance(top_n, bool):
                raise TypeError("top_n must be an int")
            if top_n < 0:
                raise ValueError("top_n must be >= 0")
        if not chunks:
            return []
        if len(chunks) == 1:
            return [_rescore(chunks[0], self._score_one(query, chunks[0]), 0)]

        pairs = [(str(query), _candidate_text(c)) for c in chunks]
        raw_scores = self._model.predict(pairs)
        scored = sorted(
            zip(raw_scores, chunks, strict=True),
            key=lambda sc: sc[0],
            reverse=True,
        )
        out: list[RetrievedChunk] = []
        for rank, (raw, chunk) in enumerate(scored):
            out.append(_rescore(chunk, self._finalize(float(raw)), rank))
        if top_n is not None:
            out = out[:top_n]
        return out

    def _score_one(self, query: str, chunk: RetrievedChunk) -> float:
        raw = float(self._model.predict([(str(query), _candidate_text(chunk))])[0])
        return self._finalize(raw)

    def _finalize(self, raw: float) -> float:
        return _stable_sigmoid(raw) if self._apply_sigmoid else raw

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model_name={self._model_name!r}, "
            f"apply_sigmoid={self._apply_sigmoid})"
        )


def _rescore(chunk: RetrievedChunk, score: float, rank: int) -> RetrievedChunk:
    return replace(chunk, score=score, rank=rank)


__all__ = [
    "DEFAULT_CROSS_ENCODER_MODEL",
    "CrossEncoderReranker",
]
