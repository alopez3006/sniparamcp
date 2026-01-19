"""Embeddings service for semantic search.

Uses sentence-transformers with lazy model loading to avoid startup cost.
Model: all-MiniLM-L6-v2 (384 dimensions, fast and efficient)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


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

            self._model = SentenceTransformer(MODEL_NAME)
            logger.info(f"Embedding model loaded: {MODEL_NAME}")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector (384 dimensions).
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts.

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
        """
        if not doc_embeddings:
            return []

        query = np.array(query_embedding)
        docs = np.array(doc_embeddings)

        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity
        similarities = np.dot(docs_norm, query_norm)
        return similarities.tolist()

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
