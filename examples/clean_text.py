"""Demo: customizable text cleaning between conversion and generation.

Cleaning is the business-dependent step. This demo shows the full offline
pipeline -- **convert -> clean -> generate** -- using deterministic fakes so it
runs with no optional dependencies and no network.

Key idea: the *same* cleaning policy can have global rules (apply to every
document) and source/filename-specific rules (apply only to PDFs, only to a
Confluence export, ...). That is how business-specific noise (watermarks, page
footers, boilerplate, PII) gets stripped only where it should.

Run with:  PYTHONPATH=src python3 examples/clean_text.py
"""

from __future__ import annotations

from sparksage import (
    ConversionResult,
    FakeLLMClient,
    IdeaBlockGenerator,
    RegexReplaceRule,
    TextCleaner,
)


def _show(label: str, text: str) -> None:
    print(f"\n--- {label} ---")
    print(repr(text))


def main() -> None:
    # Pretend these came straight out of the converter (raw, noisy Markdown).
    raw_pdf = (
        "\ufeff# Annual Report\n"
        "CONFIDENTIAL\n\n\n\n"
        "Revenue grew 12%.\n"
        "Page 1 of 5\n\n"
        "Contact: 123-45-6789"
    )
    raw_docx = "CONFIDENTIAL\n\n\n\nStrategy: expand to APAC.\x00Page 1 of 5"

    conversions = [
        ConversionResult(markdown=raw_pdf, source="docs/annual.pdf", title="Annual"),
        ConversionResult(markdown=raw_docx, source="docs/strategy.docx"),
    ]

    # 1) Build a cleaning policy: defaults (BOM, line-endings, control chars,
    #    trailing whitespace, blank-line collapse) + business-specific rules.
    cleaner = TextCleaner()
    cleaner.add(RegexReplaceRule(r"CONFIDENTIAL", ""))  # every document
    cleaner.add(RegexReplaceRule(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED]"))  # PII
    cleaner.add_for("*.pdf", RegexReplaceRule(r"Page \d+ of \d+", ""))  # PDF footers

    print("Rules for docs/annual.pdf:", [type(r).__name__ for r in cleaner.rules_for("docs/annual.pdf")])
    print("Rules for docs/strategy.docx:", [type(r).__name__ for r in cleaner.rules_for("docs/strategy.docx")])

    _show("raw pdf", raw_pdf)
    cleaned = cleaner.clean_results(conversions)
    for c in cleaned:
        _show(f"cleaned {c.source}", c.text)
        # footers stripped only on the PDF:
        if c.source.endswith(".pdf"):
            assert "Page 1 of 5" not in c.text, "PDF footer should be removed"
        else:
            assert "Page 1 of 5" in c.text, "DOCX keeps its (different) footer policy"
        assert "CONFIDENTIAL" not in c.text
        assert "[REDACTED]" in c.text or "123-45-6789" not in c.text

    # 2) Chain straight into block generation.
    fake = FakeLLMClient(
        responses=[
            """[
          {"name": "Revenue", "critical_question": "How did revenue change?",
           "trusted_answer": "Revenue grew 12%.", "tags": ["important"], "keywords": ["revenue"]},
          {"name": "Strategy", "critical_question": "What is the strategy?",
           "trusted_answer": "Expand to APAC.", "tags": ["process"], "keywords": ["strategy"]}
        ]"""
        ]
    )
    gen = IdeaBlockGenerator(fake)
    blocks = gen.generate(cleaned[0].text, source=cleaned[0].source_ref)
    print("\n--- generated blocks ---")
    for b in blocks:
        print(f"- {b.name}: {b.critical_question}  (source={b.source.uri})")


if __name__ == "__main__":
    main()
