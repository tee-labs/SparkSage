"""RestrictedPython-backed script rules: the custom-cleaning escape hatch.

Declarative rules (``RegexReplaceRule`` / ``section_extract``-style config)
cover the simple cases, but real business cleaning is often multi-branch logic
-- "keep only chapters 3-5 of the operations manual, but only for
``*报表*.xlsx``" -- that reads far more naturally as a few lines of Python than
as a JSON config. :class:`RestrictedScriptRule` is that layer: the user writes
a plain ``clean(text, source)`` function, stored as source text and compiled
through RestrictedPython into an ordinary
:class:`~sparksage.clean.rules.CleaningRule` at construction.

Safety model (a self-hosted, trusted-internal tool, so the threat is *accidental
breakage*, not a hostile attacker):

- **RestrictedPython** strips the dangerous syntax (``import``, ``eval``,
  dunder attribute access) at compile time; ``safer_getattr`` blocks ``_``-/
  ``__``-prefixed attribute lookups at runtime.
- **Builtins** are ``safe_builtins`` (``str`` / ``len`` / ``sorted`` / ... but
  no ``open`` / ``getattr`` / ``globals`` / ``__import__``).
- **Regex is time-boxed**: the ``re`` name in the sandbox is a thin wrapper
  around the ``regex`` package that forces a ``timeout`` on every call. This is
  the ReDoS fix -- stdlib ``re`` and bare ``regex`` both hold the GIL through a
  catastrophic backtrace and freeze the whole process; with a per-call
  ``timeout`` the engine aborts the match itself.
- **Wall-clock timeout + size caps** isolate one bad rule from the rest of the
  ingest; a failure logs, sets ``last_error``, and leaves the text unchanged
  rather than crashing the pipeline.

The ``RestrictedPython`` and ``regex`` packages are optional dependencies --
``pip install 'sparksage[clean-script]'`` -- imported lazily inside ``__init__``
(matching the convention of every other optional SDK, e.g.
``embed/backends/FaissVectorStore``).
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Any

_logger = logging.getLogger(__name__)

#: Regex entry points that take a ``timeout`` kwarg in the ``regex`` package.
_REGEX_FUNCS = (
    "search",
    "match",
    "fullmatch",
    "sub",
    "subn",
    "split",
    "findall",
    "finditer",
)


def _compile_restricted_policy() -> tuple[Any, ...]:
    """Lazily import RestrictedPython and build the sandbox policy.

    Returns ``(compile_restricted, safe_builtins, guards)`` so the heavy
    imports happen only when a script rule is actually constructed.
    """
    try:
        from RestrictedPython import compile_restricted, safe_builtins
        from RestrictedPython.Guards import (
            full_write_guard,
            guarded_iter_unpack_sequence,
            guarded_unpack_sequence,
            safer_getattr,
        )
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "RestrictedScriptRule requires the optional dependency; "
            "install with: pip install 'sparksage[clean-script]'"
        ) from exc
    return (
        compile_restricted,
        safe_builtins,
        (safer_getattr, full_write_guard, guarded_unpack_sequence, guarded_iter_unpack_sequence),
    )


class _TimedPattern:
    """A compiled ``regex`` pattern whose methods default to a ``timeout``."""

    def __init__(self, pattern: Any, timeout: float) -> None:
        object.__setattr__(self, "_pattern", pattern)
        object.__setattr__(self, "_timeout", timeout)

    def __getattr__(self, name: str) -> Any:
        # passthrough for pattern / flags / groupindex / scanner / ...
        return getattr(object.__getattribute__(self, "_pattern"), name)

    def _call(self, name: str, *args: Any, **kw: Any) -> Any:
        kw.setdefault("timeout", object.__getattribute__(self, "_timeout"))
        return getattr(object.__getattribute__(self, "_pattern"), name)(*args, **kw)


for _name in _REGEX_FUNCS:

    def _make(_n: str):
        def _m(self, *args: Any, **kw: Any) -> Any:
            return self._call(_n, *args, **kw)

        _m.__name__ = _n
        return _m

    setattr(_TimedPattern, _name, _make(_name))
del _name, _make


class _TimedRegex:
    """Module-shaped ``re`` stand-in forcing a timeout on every regex call.

    Presented to the sandbox as ``re`` so scripts written against stdlib ``re``
    work unchanged while every pattern search/match/sub is bounded by the
    rule's ``timeout``. Also why stdlib ``re`` is *not* exposed: it has no
    timeout and a catastrophic backtrace would hang the entire process.
    """

    def __init__(self, timeout: float) -> None:
        import regex  # noqa: PLC0415 - lazy optional dependency

        object.__setattr__(self, "_timeout", timeout)
        object.__setattr__(self, "_regex", regex)

    def compile(self, pattern: str, flags: int = 0, **kw: Any) -> _TimedPattern:
        return _TimedPattern(
            object.__getattribute__(self, "_regex").compile(pattern, flags, **kw),
            object.__getattribute__(self, "_timeout"),
        )

    def _call(self, name: str, *args: Any, **kw: Any) -> Any:
        kw.setdefault("timeout", object.__getattribute__(self, "_timeout"))
        return getattr(object.__getattribute__(self, "_regex"), name)(*args, **kw)

    def escape(self, pattern: str) -> str:
        return object.__getattribute__(self, "_regex").escape(pattern)


for _name in _REGEX_FUNCS:

    def _make(_n: str):
        def _m(self, *args: Any, **kw: Any) -> Any:
            return self._call(_n, *args, **kw)

        _m.__name__ = _n
        return _m

    setattr(_TimedRegex, _name, _make(_name))
del _name, _make


class RestrictedScriptRule:
    """A :class:`~sparksage.clean.rules.CleaningRule` defined as sandboxed source.

    The ``code`` must define ``clean(text, source=None) -> str``. It may use
    ``str``/``list``/``re``-style operations and branch on ``source`` (the
    document path/URI), so a single rule can express multi-branch, filename-aware
    logic that declarative rules cannot.

    A failed run never raises into the pipeline: the error is logged, stored in
    :attr:`last_error`, and the input text is returned unchanged (fail-open, one
    bad rule cannot break an ingest). Use the ``last_error`` field (or the API
    test endpoint) to inspect failures before enabling a rule.

    Parameters
    ----------
    code:
        RestrictedPython source defining ``clean``.
    name:
        Rule label used in logs / error messages.
    timeout:
        Wall-clock seconds per ``clean`` call (and per regex operation) before
        the rule is treated as failed.
    max_input_chars:
        Texts larger than this are skipped without invoking the script.
    max_output_chars:
        Outputs larger than this are treated as a failure.
    """

    def __init__(
        self,
        code: str,
        *,
        name: str = "restricted-script",
        timeout: float = 5.0,
        max_input_chars: int = 1_000_000,
        max_output_chars: int = 2_000_000,
    ) -> None:
        compile_restricted, safe_builtins, (getattr_, write_, unpack_, iter_unpack_) = (
            _compile_restricted_policy()
        )
        try:
            self._code = compile_restricted(code, f"<cleaning:{name}>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"restricted script '{name}' failed to compile: {exc}") from exc

        self.name = name
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.last_error: str | None = None

        self._base = {
            "__builtins__": safe_builtins,
            "_getattr_": getattr_,
            "_getitem_": lambda obj, key: obj[key],
            "_write_": write_,
            "_unpack_sequence_": unpack_,
            "_iter_unpack_sequence_": iter_unpack_,
            "_getiter_": iter,
            "re": _TimedRegex(timeout),
        }

        fn = self._extract_clean(self._code)
        try:
            nparams = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):  # pragma: no cover - defined by us above
            nparams = 1
        self._accepts_source = nparams >= 2

    def _extract_clean(self, code: Any) -> Any:
        """Exec ``code`` in a scratch namespace and return its ``clean``."""
        ns = dict(self._base)
        try:
            exec(code, ns)
        except Exception as exc:
            raise ValueError(
                f"restricted script '{self.name}' failed: {type(exc).__name__}: {exc}"
            ) from exc
        fn = ns.get("clean")
        if not callable(fn):
            raise ValueError(
                f"restricted script '{self.name}' must define a callable 'clean(text, source=None)'"
            )
        return fn

    def clean(self, text: str, source: str | None = None) -> str:
        """Run the script over ``text``; returns ``text`` unchanged on failure."""
        self.last_error = None
        if len(text) > self.max_input_chars:
            self.last_error = f"input too large: {len(text)} chars > {self.max_input_chars}"
            return text
        ns = dict(self._base)
        try:
            exec(self._code, ns)
        except Exception as exc:  # pragma: no cover - compiled bytecode should not raise
            self.last_error = f"exec failed: {type(exc).__name__}: {exc}"
            return text
        result = self._run(ns["clean"], text, source)
        if result is None:
            self.last_error = f"timed out after {self.timeout}s"
            return text
        if len(result) > self.max_output_chars:
            self.last_error = (
                f"output too large: {len(result)} chars > {self.max_output_chars}"
            )
            return text
        return result

    def _run(self, fn: Any, text: str, source: str | None) -> str | None:
        """Call ``fn`` with a wall-clock timeout; ``None`` means timed out."""
        box: dict[str, Any] = {}

        def _invoke() -> None:
            try:
                if self._accepts_source:
                    box["result"] = fn(text, source)
                else:
                    box["result"] = fn(text)
            except Exception as exc:
                box["error"] = exc

        worker = threading.Thread(target=_invoke, daemon=True, name=f"clean-{self.name}")
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            # ponytail: threads cannot be killed, so a pure-Python infinite loop
            # leaks one spinning thread (regex is never the leak -- its timeout
            # aborts the match). Subprocess isolation if that ever matters.
            _logger.warning("clean rule '%s' timed out after %.1fs", self.name, self.timeout)
            return None
        if "error" in box:
            exc = box["error"]
            self.last_error = f"{type(exc).__name__}: {exc}"
            _logger.warning("clean rule '%s' failed: %s", self.name, self.last_error)
            return text
        return box["result"]
