#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Compare OCR benchmark results across all tools.
Reads timing.json and .md files from results/ subdirectories.
Produces a benchmark_report.md with side-by-side comparison.
"""

import json
import re
import sys
from pathlib import Path

TOOLS = ["marker", "nougat", "deepseek", "paddleocr", "docstrange"]
TOOL_NAMES = {
    "marker": "marker-pdf",
    "nougat": "nougat",
    "deepseek": "deepseek-ocr",
    "paddleocr": "paddleocr",
    "docstrange": "docstrange",
}


def count_structures(text: str) -> dict:
    """Count markdown structural elements in text."""
    return {
        "headings": len(re.findall(r"^#{1,6}\s", text, re.MULTILINE)),
        "tables": len(re.findall(r"^\|.*\|$", text, re.MULTILINE))
        // 3,  # rough: 3 lines per table
        "lists": len(re.findall(r"^[\s]*[-*+]\s", text, re.MULTILINE)),
        "code_blocks": len(re.findall(r"```", text)) // 2,
        "math_inline": len(re.findall(r"\$[^$\n]+\$", text)),
        "math_block": len(re.findall(r"\$\$.*?\$\$", text, re.DOTALL)),
        "bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "links": len(re.findall(r"\[.*?\]\(.*?\)", text)),
    }


def find_pdf_subdirs(results_dir: Path) -> list[str]:
    """Find all PDF result subdirectories across tool dirs."""
    pdf_names = set()
    for tool in TOOLS:
        tool_dir = results_dir / tool
        if tool_dir.exists():
            for subdir in tool_dir.iterdir():
                if subdir.is_dir():
                    pdf_names.add(subdir.name)
    # Also handle flat structure (no PDF subdirs)
    if not pdf_names:
        pdf_names.add("")
    return sorted(pdf_names)


def analyze_tool(results_dir: Path, tool: str, pdf_name: str = "") -> dict:
    """Analyze results for a single tool and PDF."""
    if pdf_name:
        tool_dir = results_dir / tool / pdf_name
    else:
        tool_dir = results_dir / tool

    if not tool_dir.exists():
        return {"tool": tool, "pdf": pdf_name, "status": "not_run"}

    timing_file = tool_dir / "timing.json"
    if not timing_file.exists():
        return {"tool": tool, "pdf": pdf_name, "status": "no_timing"}

    meta = json.loads(timing_file.read_text())

    if meta.get("error"):
        return {
            "tool": tool,
            "pdf": pdf_name,
            "status": "error",
            "error": meta["error"],
            **meta,
        }

    # Find the .md output
    md_files = list(tool_dir.glob("*.md"))
    text = ""
    if md_files:
        text = md_files[0].read_text(encoding="utf-8")

    structures = count_structures(text)

    return {
        "tool": tool,
        "pdf": pdf_name,
        "status": "ok",
        **meta,
        "structures": structures,
        "text_preview": text[:500] if text else "",
    }


def generate_report(results_dir: Path) -> str:
    """Generate a markdown comparison report."""
    pdf_names = find_pdf_subdirs(results_dir)

    report = "# OCR Benchmark Report\n\n"
    report += f"Results directory: `{results_dir}`\n\n"

    for pdf_name in pdf_names:
        analyses = []
        for tool in TOOLS:
            analyses.append(analyze_tool(results_dir, tool, pdf_name))

        if pdf_name:
            report += f"---\n\n## Document: {pdf_name}\n\n"

        # Timing comparison table
        report += "### Timing Comparison\n\n"
        report += (
            "| Tool | Status | Total Time | Model Load | Conversion | Chars | Words |\n"
        )
        report += (
            "|------|--------|-----------|------------|------------|-------|-------|\n"
        )

        for a in analyses:
            if a["status"] == "not_run":
                report += f"| {a['tool']} | SKIPPED | - | - | - | - | - |\n"
            elif a["status"] == "error":
                report += f"| {a['tool']} | ERROR | - | - | - | - | - |\n"
            else:
                total = a.get("total_time", "?")
                model = a.get("model_load_time", "-")
                conv = a.get("conversion_time", "-")
                chars = a.get("output_chars", 0)
                words = a.get("output_words", 0)
                report += f"| {a['tool']} | OK | {total}s | {model}s | {conv}s | {chars:,} | {words:,} |\n"

        # Structure detection table
        report += "\n### Structure Detection\n\n"
        report += "| Tool | Headings | Tables | Lists | Code | Math | Bold | Links |\n"
        report += "|------|----------|--------|-------|------|------|------|-------|\n"

        for a in analyses:
            if a["status"] != "ok":
                report += f"| {a['tool']} | - | - | - | - | - | - | - |\n"
                continue
            s = a.get("structures", {})
            math = s.get("math_inline", 0) + s.get("math_block", 0)
            report += (
                f"| {a['tool']} "
                f"| {s.get('headings', 0)} "
                f"| {s.get('tables', 0)} "
                f"| {s.get('lists', 0)} "
                f"| {s.get('code_blocks', 0)} "
                f"| {math} "
                f"| {s.get('bold', 0)} "
                f"| {s.get('links', 0)} |\n"
            )

        # Output previews
        report += "\n### Output Previews (first 500 chars)\n\n"
        for a in analyses:
            if a["status"] != "ok":
                continue
            preview = a.get("text_preview", "")
            report += f"#### {a['tool']}\n\n```\n{preview}\n```\n\n"

        # Errors
        errors = [a for a in analyses if a.get("error")]
        if errors:
            report += "\n### Errors\n\n"
            for a in errors:
                report += f"- **{a['tool']}**: {a['error'][:1000]}\n"

    return report


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    report = generate_report(results_dir)

    report_path = results_dir / "benchmark_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
