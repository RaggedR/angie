"""Ingest math research papers into Cognee's knowledge graph.

Usage:
    cd ~/git/cognee
    source .venv/bin/activate
    python ingest_papers.py
"""

import asyncio
import os
from pathlib import Path

import cognee


PAPERS_DIR = Path(os.path.expanduser("~/data/arxiv-rag"))
DATASET_NAME = "q-series-papers"

# Subdirectories containing PDFs (skip chroma_db, compressed, meta-summaries)
SUBDIRS = [
    "core",
    "q-series-positivity",
    "cylindric-partitions",
    "hall-littlewood-macdonald",
    "crystal-bases",
    "qsym-p-partitions",
]


async def main():
    # Collect all PDFs
    pdfs = []
    for subdir in SUBDIRS:
        d = PAPERS_DIR / subdir
        if d.exists():
            pdfs.extend(sorted(d.glob("*.pdf")))

    print(f"Found {len(pdfs)} PDFs across {len(SUBDIRS)} subdirectories")
    for p in pdfs:
        print(f"  {p.relative_to(PAPERS_DIR)}")

    # Add papers to Cognee
    print(f"\n--- Adding papers to dataset '{DATASET_NAME}' ---")
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] Adding {pdf.name}...")
        await cognee.add(str(pdf), dataset_name=DATASET_NAME)

    # Build the knowledge graph
    print("\n--- Running cognify (entity extraction + graph construction) ---")
    print("This will call gpt-4o-mini for extraction and math-embed for embeddings.")
    print("Estimated time: 10-30 minutes depending on paper lengths.\n")
    await cognee.cognify(datasets=[DATASET_NAME])

    print("\n--- Done! ---")
    print("Knowledge graph built. You can now search with:")
    print('  await cognee.search("Bailey lemma", search_type="GRAPH_COMPLETION")')


if __name__ == "__main__":
    asyncio.run(main())
