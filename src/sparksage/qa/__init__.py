"""SparkSage end-to-end QA: query -> retrieval -> answer.

:class:`QAEngine` closes the loop -- it is the orchestrator that finally makes
SparkSage an end-to-end *question-answering* core instead of a
preprocessing+dedup library. It wires together the three right-half stages
built in this roadmap:

* :mod:`sparksage.query`  -- query understanding (intent + rewrite + sub-queries).
* :mod:`sparksage.retrieve` -- hybrid retrieval (BM25 + dense + fusion + rerank).
* :mod:`sparksage.reader` -- answer generation with grounded citations + abstention.

It owns no business logic -- every stage is a swappable protocol -- so it runs
fully offline under :class:`~sparksage.generator.FakeLLMClient` and
:class:`~sparksage.embed.FakeEmbeddingClient`. Not yet wired to the web layer;
a future ``/api/v1/query`` route will be a thin wrapper around
:meth:`QAEngine.ask`.

Example
-------
::

    from sparksage import FakeLLMClient, FakeEmbeddingClient
    from sparksage.qa import QAEngine

    engine = QAEngine(retriever=retriever, reader=reader)
    result = engine.ask("how to deploy")
    print(result.text, result.citations)
"""

from sparksage.qa.engine import (
    DEFAULT_MAX_REFINE_ITERATIONS,
    DEFAULT_MIN_RELEVANCE,
    IntentKBRouter,
    QACache,
    QAEngine,
    QAResult,
)

__all__ = [
    "DEFAULT_MAX_REFINE_ITERATIONS",
    "DEFAULT_MIN_RELEVANCE",
    "IntentKBRouter",
    "QAEngine",
    "QACache",
    "QAResult",
]
