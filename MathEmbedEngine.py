"""Embedding engine using the math-embed model (RobBobin/math-embed).

A domain-specific 768-dim embedding model fine-tuned on combinatorics /
q-series papers using knowledge-graph-guided contrastive learning on top
of allenai/specter2_base.  Achieves MRR 0.816 vs OpenAI's 0.461 on
mathematical paper retrieval benchmarks.

Requires: pip install sentence-transformers torch
"""

import asyncio
import logging
from typing import List, Optional

import numpy as np
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.shared.logging_utils import get_logger

logger = get_logger("MathEmbedEngine")


class MathEmbedEngine(EmbeddingEngine):
    DIMENSIONS = 768

    def __init__(
        self,
        model: str = "RobBobin/math-embed",
        dimensions: int = DIMENSIONS,
        max_completion_tokens: int = 512,
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_id = model
        self.dimensions = dimensions
        self.max_completion_tokens = max_completion_tokens
        self.batch_size = batch_size
        self.tokenizer = None  # fall back to word-level counting in chunker
        logger.info("Loading math-embed model: %s", model)
        self._model = SentenceTransformer(model)
        logger.info("math-embed model loaded (dim=%d)", dimensions)

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """Embed text using math-embed. Runs inference in a thread to avoid blocking asyncio."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._embed_sync, text)
        return embeddings

    def _embed_sync(self, text: List[str]) -> List[List[float]]:
        embs = self._model.encode(
            text,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        arr = np.array(embs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # L2-normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr = arr / norms
        return arr.tolist()

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_batch_size(self) -> int:
        return self.batch_size
