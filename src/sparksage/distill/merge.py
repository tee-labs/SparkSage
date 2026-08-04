"""LLM-driven merge of a near-duplicate cluster into one canonical IdeaBlock.

:class:`BlockMerger` is the Distill counterpart of
:class:`~sparksage.generator.IdeaBlockGenerator`: instead of *decomposing* free
text into many blocks, it *fuses* a cluster of near-duplicate blocks into ONE
canonical block. It reuses the exact same :class:`~sparksage.generator.LLMClient`
protocol the generator already depends on, so it is fully unit-testable with
:class:`~sparksage.generator.FakeLLMClient` and zero network access.

The merge decision is a content decision: the LLM is shown a compact digest of
the cluster's members and asked to reconcile them into a single, more complete
yet more concise block. The result is coerced through the controlled
vocabularies (see :mod:`sparksage.distill.schema`) and stamped with the Distill
lifecycle fields: ``status=ACTIVE``, ``parents`` = every member id, and
``confidence`` = the cluster's mean pairwise similarity (supplied by the caller).
"""

from __future__ import annotations

import json

from sparksage.distill.prompts import merge_messages
from sparksage.distill.schema import (
    MergeCoercionError,
    RawMergedBlock,
    coerce_merged_block,
    parse_raw_merged,
)
from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.llmutil import extract_json
from sparksage.schema.enums import BlockStatus
from sparksage.schema.ideablock import IdeaBlock
from sparksage.schema.source import SourceRef


class MergeError(RuntimeError):
    """Base error for the Distill merge step."""


class MergeEmptyResponseError(MergeError):
    """The merge LLM returned no content."""


class MergeResponseParseError(MergeError):
    """The merge model response could not be parsed as the expected JSON."""


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a possibly-noisy merge response.

    Same three-case handling as the generator: plain JSON, fenced JSON, and JSON
    embedded in prose (outermost brace match).
    """
    return extract_json(
        text,
        error_type=MergeResponseParseError,
        empty_msg="empty model response",
        lenient=True,
    )


def _promote_single(block: IdeaBlock, *, confidence: float) -> IdeaBlock:
    """Return an ACTIVE copy of ``block`` carrying the merge confidence.

    A cluster reduced to a single member is not really "merged" with anything,
    so its ``parents`` stay empty -- but it is still promoted to ACTIVE (the
    Distill lifecycle) and stamped with the cluster confidence for auditability.
    """
    return block.model_copy(
        update={
            "status": BlockStatus.ACTIVE,
            "confidence": confidence,
            "parents": [],
        }
    )


class BlockMerger:
    """Merge a cluster of near-duplicate IdeaBlocks into one canonical block.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Decouples merging from any specific SDK.
    model:
        Model name forwarded to the client (ignored by fakes). Merging is a
        compression task, so a lightweight model is usually sufficient.
    temperature:
        Sampling temperature. Low (default ``0.2``) for faithful compression.
    language:
        BCP-47 code written into the canonical block.
    strict:
        If ``True``, a malformed/oversized merge result raises
        :class:`MergeError`. If ``False`` (default), a failed merge *falls back*
        to promoting the first member (so a single bad LLM call never aborts a
        whole 100k-block Distill run).
    use_json_mode:
        Request JSON-mode structured output from the provider when supported.

    Examples
    --------
    >>> from sparksage import FakeLLMClient
    >>> from sparksage.distill import BlockMerger
    >>> merger = BlockMerger(FakeLLMClient())   # doctest: +SKIP
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        language: str = "en",
        strict: bool = False,
        use_json_mode: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._language = language
        self._strict = strict
        self._use_json_mode = use_json_mode
        self.merge_calls: int = 0
        self.fallbacks: int = 0

    @property
    def strict(self) -> bool:
        return self._strict

    def merge_cluster(
        self,
        blocks: list[IdeaBlock],
        *,
        confidence: float = 1.0,
        source: SourceRef | None = None,
    ) -> IdeaBlock:
        """Fuse ``blocks`` into one canonical, ACTIVE :class:`IdeaBlock`.

        The canonical block carries ``parents`` = the UUIDs of every member, and
        ``confidence`` = ``confidence`` (the cluster's mean pairwise similarity,
        supplied by the pipeline). A single-element cluster is a no-op merge:
        the sole member is promoted to ACTIVE with empty ``parents``.

        Parameters
        ----------
        blocks:
            The near-duplicate members. Must be non-empty.
        confidence:
            Cluster confidence written to the canonical block, in ``[0, 1]``.
        source:
            Provenance for the canonical block. When ``None``, the most specific
            source among the members is inherited (preferring a member source
            over nothing).

        Raises
        ------
        MergeError
            If the LLM call fails outright, or in ``strict`` mode if the result
            cannot be coerced. In non-strict mode a coercion failure falls back
            to promoting the first member rather than raising.
        """
        if not blocks:
            raise ValueError("merge_cluster requires at least one block")

        if len(blocks) == 1:
            return _promote_single(blocks[0], confidence=confidence)

        parent_ids = [b.id for b in blocks]
        inherited_source = source if source is not None else _inherit_source(blocks)

        self.merge_calls += 1
        response_text = self._client.complete(
            merge_messages(blocks),
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            if self._strict:
                raise MergeEmptyResponseError("the merge LLM returned an empty response")
            self.fallbacks += 1
            return self._fallback(blocks, confidence, inherited_source)

        try:
            raw = self._parse(response_text)
            return coerce_merged_block(
                raw,
                parents=parent_ids,
                confidence=confidence,
                source=inherited_source,
                language=self._language,
                strict=self._strict,
            )
        except (MergeCoercionError, MergeResponseParseError) as exc:
            if self._strict:
                raise MergeError(str(exc)) from exc
            self.fallbacks += 1
            return self._fallback(blocks, confidence, inherited_source)

    def _parse(self, response_text: str) -> RawMergedBlock:
        payload = _extract_json(response_text)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MergeResponseParseError(
                f"merge response was not valid JSON: {exc.msg}"
            ) from exc
        try:
            return parse_raw_merged(data)
        except MergeCoercionError as exc:
            raise MergeResponseParseError(str(exc)) from exc

    def _fallback(
        self,
        blocks: list[IdeaBlock],
        confidence: float,
        source: SourceRef | None,
    ) -> IdeaBlock:
        """Non-strict fallback: promote the first member as the canonical.

        Keeps the Distill run resilient -- one bad LLM merge must not abort a
        large corpus. The promoted block still records the cluster as its
        parents and carries the confidence, so no information is lost.
        """
        canonical = blocks[0].model_copy(
            update={
                "status": BlockStatus.ACTIVE,
                "confidence": confidence,
                "parents": [b.id for b in blocks[1:]],
                "source": source if source is not None else blocks[0].source,
            }
        )
        return canonical


def _inherit_source(blocks: list[IdeaBlock]) -> SourceRef | None:
    """Pick the most specific :class:`SourceRef` among ``blocks`` (first non-null)."""
    for block in blocks:
        if block.source is not None:
            return block.source
    return None


__all__ = [
    "BlockMerger",
    "MergeEmptyResponseError",
    "MergeError",
    "MergeResponseParseError",
]
