"""Run cognify with custom math extraction prompt.

Usage:
    python run_cognify.py                    # All papers
    python run_cognify.py --test             # Single paper (Warnaar A2 Andrews-Gordon)
    python run_cognify.py --test --fresh     # Wipe data first, then single paper
"""

import asyncio
import sys
from pathlib import Path

import cognee

DATASET_NAME = "q-series-papers"
PROMPT_PATH = Path(__file__).parent / "math_graph_prompt.txt"
PAPERS_DIR = Path.home() / "data" / "arxiv-rag"
TEST_PAPER = PAPERS_DIR / "core" / "warnaar_2021_A2_andrews_gordon.pdf"

SUBDIRS = [
    "core", "q-series-positivity", "cylindric-partitions",
    "hall-littlewood-macdonald", "crystal-bases", "qsym-p-partitions",
]


async def main():
    test_mode = "--test" in sys.argv
    fresh = "--fresh" in sys.argv

    math_prompt = PROMPT_PATH.read_text()

    if fresh:
        print("--- Wiping existing data ---")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system()

    if test_mode:
        print(f"--- TEST MODE: single paper ---")
        print(f"Paper: {TEST_PAPER.name}")
        print(f"Prompt: {PROMPT_PATH.name} ({len(math_prompt)} chars)")

        await cognee.add(str(TEST_PAPER), dataset_name=DATASET_NAME)

        print("\n--- Running cognify ---")
        await cognee.cognify(
            datasets=[DATASET_NAME],
            custom_prompt=math_prompt,
            chunk_size=16000,
        )
    else:
        # Collect all PDFs
        pdfs = []
        for subdir in SUBDIRS:
            d = PAPERS_DIR / subdir
            if d.exists():
                pdfs.extend(sorted(d.glob("*.pdf")))

        print(f"Papers: {len(pdfs)}")
        print(f"Prompt: {PROMPT_PATH.name} ({len(math_prompt)} chars)")

        print(f"\n--- Adding papers ---")
        for i, pdf in enumerate(pdfs, 1):
            print(f"[{i}/{len(pdfs)}] {pdf.name}")
            await cognee.add(str(pdf), dataset_name=DATASET_NAME)

        print("\n--- Running cognify ---")
        await cognee.cognify(
            datasets=[DATASET_NAME],
            custom_prompt=math_prompt,
            chunk_size=16000,
        )

    print("\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())
