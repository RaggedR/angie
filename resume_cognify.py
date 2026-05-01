"""Resume cognify on already-ingested papers."""

import asyncio
from pathlib import Path
import cognee

DATASET_NAME = "q-series-papers"
PROMPT_PATH = Path(__file__).parent / "math_graph_prompt.txt"


async def main():
    math_prompt = PROMPT_PATH.read_text()
    print(f"Prompt: {len(math_prompt)} chars")
    print("Resuming cognify (5 req/min rate limit, 16K chunks)...\n")

    await cognee.cognify(
        datasets=[DATASET_NAME],
        custom_prompt=math_prompt,
        chunk_size=16000,
    )
    print("\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())
