"""Embeddings service for semantic search.

Uses sentence-transformers with lazy model loading to avoid startup cost.
Model: BAAI/bge-large-en-v1.5 (1024 dimensions, high quality)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024

# Timeout configuration (in seconds)
# Note: First request may take longer due to model cold-start (loading into memory)
EMBEDDING_TIMEOUT_SINGLE = 30.0  # Timeout for single text embedding (includes cold-start)
EMBEDDING_TIMEOUT_BATCH = 60.0  # Timeout for batch embedding (larger texts)

# Thread pool for embedding operations
_embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")


class EmbeddingsService:
    """Service for generating text embeddings using sentence-transformers.

    Uses lazy loading to avoid loading the model until first use,
    which improves startup time for requests that don't need embeddings.
    """

    _model: SentenceTransformer | None = None
    _instance: EmbeddingsService | None = None

    def __init__(self) -> None:
        """Initialize the embeddings service (model loaded lazily)."""
        pass

    @classmethod
    def get_instance(cls) -> EmbeddingsService:
        """Get singleton instance of the embeddings service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model (lazy loading).

        Returns:
            The loaded SentenceTransformer model.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            # Import here to avoid loading torch at startup
            from sentence_transformers import SentenceTransformer

            # Explicitly load on CPU to avoid meta tensor issues on memory-constrained environments
            self._model = SentenceTransformer(MODEL_NAME, device="cpu")
            logger.info(f"Embedding model loaded: {MODEL_NAME} (device: cpu)")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text (synchronous).

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector (1024 dimensions).
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts (synchronous).

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    async def embed_text_async(
        self, text: str, timeout: float = EMBEDDING_TIMEOUT_SINGLE
    ) -> list[float]:
        """Generate embedding for a single text with timeout protection.

        Args:
            text: The text to embed.
            timeout: Maximum time to wait for embedding (default: 5 seconds).

        Returns:
            List of floats representing the embedding vector (1024 dimensions).

        Raises:
            asyncio.TimeoutError: If embedding takes longer than timeout.
        """
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(_embedding_executor, self.embed_text, text),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Embedding timed out after {timeout}s for text of length {len(text)}")
            raise

    async def embed_texts_async(
        self, texts: list[str], batch_size: int = 32, timeout: float = EMBEDDING_TIMEOUT_BATCH
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts with timeout protection.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once.
            timeout: Maximum time to wait for embeddings (default: 30 seconds).

        Returns:
            List of embedding vectors.

        Raises:
            asyncio.TimeoutError: If embedding takes longer than timeout.
        """
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _embedding_executor,
                    lambda: self.embed_texts(texts, batch_size),
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"Batch embedding timed out after {timeout}s for {len(texts)} texts"
            )
            raise

    def cosine_similarity(
        self,
        query_embedding: list[float],
        doc_embeddings: list[list[float]]
    ) -> list[float]:
        """Calculate cosine similarity between query and document embeddings.

        Args:
            query_embedding: The query embedding vector.
            doc_embeddings: List of document embedding vectors.

        Returns:
            List of similarity scores (0 to 1).

        Raises:
            ValueError: If embeddings have inconsistent dimensions.
        """
        if not doc_embeddings:
            return []

        try:
            query = np.array(query_embedding)
            expected_dim = len(query_embedding)

            # Filter out embeddings with mismatched dimensions
            valid_embeddings = []
            invalid_indices = []
            for idx, emb in enumerate(doc_embeddings):
                if len(emb) == expected_dim:
                    valid_embeddings.append(emb)
                else:
                    invalid_indices.append(idx)

            if invalid_indices:
                logger.warning(
                    f"Filtered {len(invalid_indices)} embeddings with wrong dimensions "
                    f"(expected {expected_dim}): indices {invalid_indices[:10]}"
                )

            if not valid_embeddings:
                return [0.0] * len(doc_embeddings)

            docs = np.array(valid_embeddings)

            # Validate dimensions
            if docs.ndim != 2:
                raise ValueError(
                    f"Document embeddings must be 2-dimensional, got shape {docs.shape}. "
                    "This indicates inconsistent embedding dimensions."
                )

            # Normalize vectors
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)

            # Cosine similarity
            valid_similarities = np.dot(docs_norm, query_norm).tolist()

            # Reconstruct full result list with 0.0 for invalid embeddings
            if invalid_indices:
                invalid_set = set(invalid_indices)
                result = [0.0] * len(doc_embeddings)
                valid_idx = 0
                for i in range(len(doc_embeddings)):
                    if i not in invalid_set:
                        result[i] = valid_similarities[valid_idx]
                        valid_idx += 1
                return result

            return valid_similarities
        except ValueError as e:
            # Re-raise ValueError with context
            logger.error(f"Cosine similarity dimension error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in cosine similarity: {e}")
            raise ValueError(f"Failed to calculate cosine similarity: {e}") from e

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return MODEL_NAME

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None


def get_embeddings_service() -> EmbeddingsService:
    """Get the singleton embeddings service instance."""
    return EmbeddingsService.get_instance()
