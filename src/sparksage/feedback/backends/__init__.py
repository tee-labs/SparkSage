"""Concrete durable backends for the feedback layer.

* :class:`SqliteFeedbackStore` -- durable single-file persistence for
  :class:`~sparksage.feedback.models.FeedbackRecord` over a stdlib ``sqlite3``
  connection. The feedback counterpart of
  :class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`.
"""

from __future__ import annotations

from sparksage.feedback.backends.sqlite import SqliteFeedbackStore

__all__ = ["SqliteFeedbackStore"]
