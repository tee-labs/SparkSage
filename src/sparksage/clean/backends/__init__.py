"""Concrete durable backends for the cleaning-rule layer."""

from __future__ import annotations

from sparksage.clean.backends.sqlite import SqliteCleaningRuleStore

__all__ = ["SqliteCleaningRuleStore"]
