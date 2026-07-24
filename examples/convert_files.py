"""Demo: convert heterogeneous source files into Markdown.

Two modes are shown:

1. **Offline** (default) using :class:`FakeConverterBackend` -- no ``markitdown``
   installation or network needed.
2. **Real** using :class:`MarkItDownBackend` when ``--real`` is passed and the
   optional ``convert`` extra is installed::

       pip install 'sparksage[convert]'

Run with:  PYTHONPATH=src python3 examples/convert_files.py
           PYTHONPATH=src python3 examples/convert_files.py --real path/to/docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sparksage import (
    ConversionResult,
    FakeConverterBackend,
    MarkdownConverter,
)


def _print_results(results: list[ConversionResult]) -> None:
    print(f"Converted {len(results)} file(s):\n")
    for r in results:
        print(f"--- {Path(r.source).name}  (title={r.title!r}) ---")
        body = r.markdown if len(r.markdown) <= 300 else r.markdown[:297] + "..."
        print(body)
        print(f"-> source_ref.uri = {r.source_ref.uri}\n")


def _demo_offline() -> None:
    fake = FakeConverterBackend(
        markdown="# Converted document\n\nThis file was normalized to Markdown.",
        title="Sample",
    )
    conv = MarkdownConverter(backend=fake)
    results = [conv.convert("report.pdf"), conv.convert("spreadsheet.xlsx")]
    _print_results(results)


def _demo_real(path: str) -> None:
    src = Path(path)
    conv = MarkdownConverter()
    if src.is_dir():
        results = conv.convert_directory(src)
    else:
        results = [conv.convert(str(src))]
    _print_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="SparkSage file-to-Markdown demo")
    parser.add_argument(
        "--real",
        metavar="PATH",
        help="Convert a real file/directory with MarkItDownBackend (needs 'convert' extra).",
    )
    args = parser.parse_args()
    if args.real:
        _demo_real(args.real)
    else:
        _demo_offline()


if __name__ == "__main__":
    main()
