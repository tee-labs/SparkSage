"""Enumerations for the SparkSage chunk schema.

These controlled vocabularies keep knowledge chunks consistent across sources,
which makes filtering, hybrid retrieval, and provenance tracking reliable.
"""

from __future__ import annotations

from enum import Enum


class Tag(str, Enum):
    """Coarse semantic tags attached to an :class:`IdeaBlock`.

    Tags are upper-cased free-form by convention, but a shared vocabulary lets
    retrieval layers filter results (e.g. only surface ``IMPORTANT`` blocks)
    and lets admins enforce access/permission policies.
    """

    IMPORTANT = "IMPORTANT"
    WARNING = "WARNING"
    TECHNOLOGY = "TECHNOLOGY"
    PROCESS = "PROCESS"
    REFERENCE = "REFERENCE"
    FAQ = "FAQ"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    SECURITY = "SECURITY"
    ARCHITECTURE = "ARCHITECTURE"
    API = "API"
    DATASET = "DATASET"
    POLICY = "POLICY"


class EntityType(str, Enum):
    """The type of a real-world thing referenced by a chunk entity.

    A closed set keeps entity graphs queryable and avoids the "same thing,
    many spellings" problem that plagues naive metadata.
    """

    PRODUCT = "PRODUCT"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    SERVICE = "SERVICE"
    COMPONENT = "COMPONENT"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    DATASET = "DATASET"
    DOCUMENT = "DOCUMENT"
    METRIC = "METRIC"


class EntityRelation(str, Enum):
    """How an :class:`Entity` relates to the chunk that references it."""

    SUBJECT = "SUBJECT"
    OBJECT = "OBJECT"
    MENTIONS = "MENTS"
    DEPENDS_ON = "DEPENDS_ON"
    PRODUCES = "PRODUCES"
    CONSUMES = "CONSUMES"
    PART_OF = "PART_OF"
    OWNS = "OWNS"


class SentenceRole(str, Enum):
    """Role of a single sentence inside an ordered (Technical) block.

    Technical/operational content (manuals, SOPs, runbooks) is order-sensitive
    and mixes statement types. Tagging each sentence's role preserves that
    structure so retrieval can target, e.g., only ``COMMAND`` steps.
    """

    INFO = "INFO"
    COMMAND = "COMMAND"
    WARNING = "WARNING"
    PREREQUISITE = "PREREQUISITE"
    REFERENCE = "REFERENCE"
    RESULT = "RESULT"


class BlockStatus(str, Enum):
    """Lifecycle status of a block, used by the Distill dedup pipeline.

    - ``ACTIVE``: live block eligible for retrieval.
    - ``MERGED``: superseded into a parent block after de-duplication; kept for
      audit/rollback, hidden from retrieval by default.
    - ``DRAFT``: ingested but not yet validated/distilled.
    - ``ARCHIVED``: retired from the live index.
    """

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    MERGED = "MERGED"
    ARCHIVED = "ARCHIVED"


class TagSource(str, Enum):
    """How a document's tags were populated.

    Document tags are free-form strings (unlike the controlled
    :class:`IdeaBlock` :class:`Tag` vocabulary), because the knowledge-management
    workflow lets users tag freely *and* lets the system auto-generate tags from
    content. This enum records which path produced the current tag set so the
    management UI can show provenance and decide whether to re-run extraction.
    """

    USER = "USER"
    AUTO = "AUTO"
    MIXED = "MIXED"
