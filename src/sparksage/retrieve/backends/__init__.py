"""Concrete :class:`~sparksage.retrieve.reranker.Reranker` backends.

These are the production re-ranking backends that complement the
dependency-free :class:`~sparksage.retrieve.IdentityReranker` /
:class:`~sparksage.retrieve.LLMReranker`. Each implements the
:class:`~sparksage.retrieve.Reranker` protocol and lazily imports its own SDK,
so the SparkSage retrieval core stays zero-dependency -- install only the
backend you need:

* :class:`CrossEncoderReranker` -- a bi-encoder / cross-encoder model (e.g.
  ``cross-encoder/ms-marco-MiniLM-L-6-v2`` or the multilingual
  ``BAAI/bge-reranker-v2-m3``) re-scoring the fused candidate pool with one
  query/document cross-attention pass (``pip install 'sparksage[rerank]'``).

A cross-encoder is, after chunking strategy, the single largest point lever on
RAG answer quality, and is typically an order of magnitude cheaper per query
than :class:`~sparksage.retrieve.LLMReranker` while producing more precise
rankings. Swap it in anywhere a :class:`~sparksage.retrieve.Reranker` is
accepted (e.g. :class:`~sparksage.retrieve.Retriever`).
"""

from __future__ import annotations

from sparksage.retrieve.backends.cross_encoder import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
)

__all__ = [
    "DEFAULT_CROSS_ENCODER_MODEL",
    "CrossEncoderReranker",
]
