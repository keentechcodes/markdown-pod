#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
# ]
# ///
"""
Generate visual comparison charts from OCR benchmark results.

Usage:
    uv run visualize_results.py <results_dir> [output_dir]

Outputs:
    - timing_comparison.png    - Bar chart of processing times
    - content_extraction.png   - Bar chart of chars/words extracted
    - structure_detection.png  - Grouped bar chart of structural elements
    - radar_comparison.png     - Radar/spider chart for overall comparison
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np

TOOLS = ["marker", "nougat", "deepseek", "paddleocr", "docstrange"]
TOOL_LABELS = {
    "marker": "marker-pdf",
    "nougat": "nougat",
    "deepseek": "deepseek-ocr",
    "paddleocr": "paddleocr",
    "docstrange": "docstrange",
}
TOOL_COLORS = {
    "marker": "#3498db",
    "nougat": "#9b59b6",
    "deepseek": "#e74c3c",
    "paddleocr": "#f39c12",
    "docstrange": "#2ecc71",
}


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    results = {}
    for tool in TOOLS:
        tool_dir = results_dir / tool
        if not tool_dir.exists():
            continue

        for subdir in tool_dir.iterdir():
            if subdir.is_dir():
                timing_file = subdir / "timing.json"
                if timing_file.exists():
                    data = json.loads(timing_file.read_text())
                    if "error" not in data or not data.get("error"):
                        results[tool] = data
                        break
    return results


def plot_timing(results: Dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    tools = []
    times = []
    colors = []

    for tool in TOOLS:
        if tool in results:
            tools.append(TOOL_LABELS[tool])
            times.append(results[tool].get("total_time", 0))
            colors.append(TOOL_COLORS[tool])

    bars = ax.bar(tools, times, color=colors, edgecolor="white", linewidth=1.5)

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_xlabel("OCR Tool", fontsize=12)
    ax.set_title("OCR Processing Time Comparison", fontsize=14, fontweight="bold")

    for bar, time in zip(bars, times):
        ax.annotate(
            f"{time:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "timing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: timing_comparison.png")


def plot_content(results: Dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(TOOLS))
    width = 0.35

    tools_present = [t for t in TOOLS if t in results]
    x = np.arange(len(tools_present))

    chars = [results[t].get("output_chars", 0) / 1000 for t in tools_present]
    words = [results[t].get("output_words", 0) / 1000 for t in tools_present]
    colors = [TOOL_COLORS[t] for t in tools_present]

    bars1 = ax.bar(
        x - width / 2,
        chars,
        width,
        label="Characters (K)",
        color=[c for c in colors],
        alpha=0.8,
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        words,
        width,
        label="Words (K)",
        color=[c for c in colors],
        alpha=0.5,
        edgecolor="white",
        hatch="///",
    )

    ax.set_ylabel("Count (thousands)", fontsize=12)
    ax.set_xlabel("OCR Tool", fontsize=12)
    ax.set_title("Content Extraction Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TOOL_LABELS[t] for t in tools_present])
    ax.legend(loc="upper right")

    for bar, val in zip(bars1, chars):
        ax.annotate(
            f"{val:.1f}K",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, words):
        ax.annotate(
            f"{val:.1f}K",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "content_extraction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: content_extraction.png")


def plot_structure(results: Dict, md_files: Dict, output_dir: Path):
    import re

    def count_structures(text: str) -> dict:
        return {
            "headings": len(re.findall(r"^#{1,6}\s", text, re.MULTILINE)),
            "tables": len(re.findall(r"^\|.*\|$", text, re.MULTILINE)) // 3,
            "lists": len(re.findall(r"^[\s]*[-*+]\s", text, re.MULTILINE)),
            "bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
            "links": len(re.findall(r"\[.*?\]\(.*?\)", text)),
        }

    structures = {}
    for tool, md_path in md_files.items():
        if md_path and md_path.exists():
            text = md_path.read_text(encoding="utf-8")
            structures[tool] = count_structures(text)

    if not structures:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    tools_present = list(structures.keys())
    metrics = ["headings", "lists", "bold", "links"]
    metric_labels = ["Headings", "Lists", "Bold", "Links"]

    x = np.arange(len(metrics))
    width = 0.15

    for i, tool in enumerate(tools_present):
        values = [structures[tool].get(m, 0) for m in metrics]
        offset = (i - len(tools_present) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=TOOL_LABELS[tool],
            color=TOOL_COLORS[tool],
            alpha=0.85,
        )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Structure Type", fontsize=12)
    ax.set_title("Document Structure Detection", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="upper right", ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "structure_detection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: structure_detection.png")


def plot_radar(results: Dict, md_files: Dict, output_dir: Path):
    import re

    def get_scores(tool: str, data: dict, text: str) -> dict:
        chars = data.get("output_chars", 0)
        time = data.get("total_time", 1)

        return {
            "Speed": min(100, 5000 / time),
            "Content": min(100, chars / 600),
            "Headings": min(100, len(re.findall(r"^#{1,6}\s", text, re.MULTILINE)) * 2),
            "Lists": min(
                100, len(re.findall(r"^[\s]*[-*+]\s", text, re.MULTILINE)) * 1.5
            ),
            "Bold": min(100, len(re.findall(r"\*\*[^*]+\*\*", text)) * 1.5),
            "Tables": min(
                100, len(re.findall(r"^\|.*\|$", text, re.MULTILINE)) // 3 * 15
            ),
        }

    scores = {}
    for tool in TOOLS:
        if tool in results and tool in md_files and md_files[tool]:
            text = (
                md_files[tool].read_text(encoding="utf-8")
                if md_files[tool].exists()
                else ""
            )
            scores[tool] = get_scores(tool, results[tool], text)

    if len(scores) < 3:
        return

    categories = list(next(iter(scores.values())).keys())
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for tool, tool_scores in scores.items():
        values = list(tool_scores.values())
        values += values[:1]

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=TOOL_LABELS[tool],
            color=TOOL_COLORS[tool],
        )
        ax.fill(angles, values, alpha=0.15, color=TOOL_COLORS[tool])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)

    ax.set_title("OCR Tool Comparison Radar", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: radar_comparison.png")


def plot_speed_vs_quality(results: Dict, md_files: Dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 7))

    for tool in TOOLS:
        if tool not in results:
            continue

        time = results[tool].get("total_time", 0)
        chars = results[tool].get("output_chars", 0)

        ax.scatter(
            time,
            chars,
            s=300,
            c=TOOL_COLORS[tool],
            label=TOOL_LABELS[tool],
            edgecolors="white",
            linewidth=2,
            alpha=0.85,
        )

        ax.annotate(
            TOOL_LABELS[tool],
            (time, chars),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Processing Time (seconds)", fontsize=12)
    ax.set_ylabel("Characters Extracted", fontsize=12)
    ax.set_title("Speed vs Content Extraction", fontsize=14, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.7)

    top = max(r.get("output_chars", 0) for r in results.values()) * 1.1
    ax.axhline(
        y=top * 0.8, color="green", linestyle="--", alpha=0.3, label="High quality"
    )
    ax.axvline(x=60, color="blue", linestyle="--", alpha=0.3, label="Fast (<1min)")

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: speed_vs_quality.png")


def find_md_files(results_dir: Path, results: Dict) -> Dict[str, Path]:
    md_files = {}
    for tool in TOOLS:
        if tool not in results:
            continue
        tool_dir = results_dir / tool
        if tool_dir.exists():
            for subdir in tool_dir.iterdir():
                if subdir.is_dir():
                    md_list = list(subdir.glob("*.md"))
                    if md_list:
                        md_files[tool] = md_list[0]
                        break
    return md_files


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run visualize_results.py <results_dir> [output_dir]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {results_dir}")
    results = load_results(results_dir)

    if not results:
        print("No valid results found.")
        sys.exit(1)

    print(f"Found results for: {', '.join(TOOL_LABELS[t] for t in results.keys())}")

    md_files = find_md_files(results_dir, results)

    print(f"\nGenerating visualizations...")
    plot_timing(results, output_dir)
    plot_content(results, output_dir)
    plot_structure(results, md_files, output_dir)
    plot_radar(results, md_files, output_dir)
    plot_speed_vs_quality(results, md_files, output_dir)

    print(f"\nDone! Charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
