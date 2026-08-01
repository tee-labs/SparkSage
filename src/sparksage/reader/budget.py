"""Token-budget trimming for reader context (Context-Cliff protection).

Empirical "lost in the middle" studies show LLM answer quality degrades once
the stuffed context crosses roughly 2500-3000 tokens: the model starts dropping
information in the middle of the prompt. The reader previously forwarded *all*
retrieved chunks to the answer generator, so a generous top-k could quietly hurt
the very answer quality retrieval worked to improve.

:func:`trim_to_token_budget` is the cheap guard: given the best-first
:class:`~sparksage.retrieve.RetrievedChunk` list (the
:class:`~sparksage.retrieve.Retriever` contract), keep a prefix that fits a token
budget while always retaining at least ``keep_min`` chunks (so a single
over-long chunk never starves the generator of all context). It is pure stdlib
and uses the same ``len / chars_per_token`` heuristic as
:func:`sparksage.bench.metrics.approx_tokens` -- duplicated here (rather than
imported) so the *core* reader package does not depend on the *benchmark* tool
layer (which itself depends on the core). Plug in a real tokenizer via
``token_counter`` for exact budgets.

The :class:`~sparksage.reader.Reader` applies this before both generation and
faithfulness judging, so the judge scores the answer against the *same* context
the generator actually saw.

:func:`reorder_head_tail` is the complementary lost-in-the-middle guard: after
trimming it interleaves the best-first list so the strongest chunks sit at the
head *and* tail (where LLMs attend most), rather than only at the head. The
:class:`~sparksage.reader.Reader` applies it optionally (``reorder_context=``).
"""

from __future__ import annotations

from collections.abc import Callable

from sparksage.retrieve.models import RetrievedChunk

#: Default ``chars_per_token`` heuristic (the standard OpenAI-ish ``4``).
DEFAULT_CHARS_PER_TOKEN: float = 4.0

#: Default minimum chunks retained regardless of budget (see :func:`trim_to_token_budget`).
DEFAULT_KEEP_MIN: int = 1


def approx_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> float:
    """Cheap token estimate (``len / chars_per_token``), no tokenizer needed.

    Mirrors :func:`sparksage.bench.metrics.approx_tokens` so the reader and the
    benchmark agree on the heuristic without the core importing the tool layer.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    return len(text) / chars_per_token


def _candidate_text(chunk: RetrievedChunk) -> str:
    """The QA-aligned text whose token cost we account for.

    Matches what the answer prompt actually renders per candidate
    (``critical_question`` + ``trusted_answer``) -- the two fields that dominate
    the prompt's size.
    """
    block = chunk.block
    return f"{block.critical_question}\n{block.trusted_answer}"


def trim_to_token_budget(
    chunks: list[RetrievedChunk],
    max_tokens: float,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    token_counter: Callable[[str], int] | None = None,
    keep_min: int = DEFAULT_KEEP_MIN,
) -> list[RetrievedChunk]:
    """Return the best-first prefix of ``chunks`` that fits ``max_tokens``.

    Parameters
    ----------
    chunks:
        Candidate chunks in best-first order (the
        :class:`~sparksage.retrieve.Retriever` / reranker contract). The
        function does *not* re-sort: a wrong input order yields a wrong prefix.
    max_tokens:
        Soft upper bound on the total token cost of the returned chunks.
    chars_per_token:
        Heuristic ratio for the built-in estimator (ignored when
        ``token_counter`` is given).
    token_counter:
        Optional exact tokenizer (``str -> int`` token count). When ``None``
        the ``len / chars_per_token`` estimator is used.
    keep_min:
        Minimum number of chunks to retain even when the very first chunk
        already exceeds the budget (default ``1``) -- a starving context is
        worse than a slightly over-budget one. Must be ``>= 1``.

    The returned list is a new list (the input is not mutated); chunks are kept
    in input order. When ``chunks`` is empty an empty list is returned
    regardless of ``keep_min``.
    """
    if not isinstance(max_tokens, (int, float)) or isinstance(max_tokens, bool):
        raise TypeError("max_tokens must be a number")
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    if not isinstance(keep_min, int) or isinstance(keep_min, bool):
        raise TypeError("keep_min must be an int")
    if keep_min < 1:
        raise ValueError("keep_min must be >= 1")

    if not chunks:
        return []

    def count(text: str) -> float:
        if token_counter is None:
            return approx_tokens(text, chars_per_token)
        return float(token_counter(text))

    kept: list[RetrievedChunk] = []
    used = 0.0
    for chunk in chunks:
        cost = count(_candidate_text(chunk))
        if len(kept) >= keep_min and used + cost > max_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept


def reorder_head_tail(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder a best-first list so the strongest chunks sit at head and tail.

    "Lost in the middle" studies (Liu et al., 2023) show LLMs attend most to the
    *beginning* and *end* of a stuffed context and drop information in the
    middle. :func:`trim_to_token_budget` keeps the best-first prefix but leaves
    the strongest chunk at the head only. This function interleaves so the most
    relevant chunks land at **both** ends, sandwiching weaker ones in the middle
    where they are least harmful.

    The best-first list ``[a, b, c, d, e, f]`` (``a`` strongest) becomes
    ``[a, c, e, f, d, b]``: even-ranked items keep their order at the head, the
    odd-ranked items are reversed and appended so the second-strongest (``b``)
    lands at the very tail. The function is a pure permutation (no chunk is
    dropped, no score changed); it is idempotent-free by design -- apply it once
    after trimming. Empty / single-element lists are returned unchanged.
    """
    if len(chunks) < 2:
        return list(chunks)
    front = chunks[0::2]
    back = chunks[1::2]
    return [*front, *reversed(back)]


__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_KEEP_MIN",
    "approx_tokens",
    "reorder_head_tail",
    "trim_to_token_budget",
]
