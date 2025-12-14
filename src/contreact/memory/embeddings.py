"""Configurable embedding providers for memory similarity detection."""

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import requests


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of embedding vector (float32)
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Number of dimensions in embedding vectors
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Get model identifier.

        Returns:
            Model name/identifier string
        """
        pass


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """API-based embeddings via OpenRouter.

    Supports:
    - openai/text-embedding-3-small ($0.02/M tokens, 1536 dims)
    - openai/text-embedding-3-large ($0.13/M tokens, 3072 dims)
    - google/gemini-embedding-001 ($0.15/M tokens, 768 dims)
    """

    # Model dimensions (hardcoded for performance)
    MODEL_DIMENSIONS = {
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-3-large": 3072,
        "google/gemini-embedding-001": 768,
    }

    def __init__(self, model: str, api_key: Optional[str] = None):
        """Initialize OpenRouter embedding provider.

        Args:
            model: Model identifier (e.g., "openai/text-embedding-3-small")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)

        Raises:
            ValueError: If API key not provided and not in environment
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)  # Default to 1536

    def encode(self, text: str) -> np.ndarray:
        """Generate embedding via OpenRouter API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as float32 numpy array

        Raises:
            requests.HTTPError: If API request fails

        Note:
            OpenRouter embeddings endpoint may require verification.
            Alternative: Use OpenAI API directly for embeddings.
        """
        # TODO: Verify OpenRouter embeddings endpoint
        # May need to use OpenAI API directly: https://api.openai.com/v1/embeddings
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/contreact",  # Required by OpenRouter
                "X-Title": "ContReAct Memory System",
            },
            json={"model": self.model, "input": text},
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        # Extract embedding from response
        embedding = data["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)

    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def model_name(self) -> str:
        """Get model identifier."""
        return self.model


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embeddings via sentence-transformers.

    Requires optional dependencies:
        pip install sentence-transformers

    Supports:
    - all-MiniLM-L6-v2 (384 dims, 80MB, fast)
    - all-mpnet-base-v2 (768 dims, 420MB, better quality)
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """Initialize local embedding provider.

        Args:
            model: SentenceTransformer model name

        Note:
            Model is loaded lazily on first encode() call
        """
        self.model_name_str = model
        self._model = None  # Lazy load
        self._dimension_cache = None

    def _ensure_model(self):
        """Lazy load sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

            print(f"Loading local embedding model: {self.model_name_str}...")
            self._model = SentenceTransformer(self.model_name_str)
            print(f"Model loaded ({self.dimension()} dimensions)")

    def encode(self, text: str) -> np.ndarray:
        """Generate embedding using local model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as float32 numpy array
        """
        self._ensure_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension_cache is None:
            self._ensure_model()
            self._dimension_cache = self._model.get_sentence_embedding_dimension()
        return self._dimension_cache

    def model_name(self) -> str:
        """Get model identifier."""
        return self.model_name_str


def create_embedding_provider(config: dict) -> EmbeddingProvider:
    """Factory function to create embedding provider from config.

    Args:
        config: Configuration dict with keys:
            - provider: "openrouter" or "local" (default: "local")
            - model: Model identifier
            - api_key: (optional) API key for OpenRouter

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider unknown or required fields missing

    Example:
        >>> config = {
        ...     "provider": "local",
        ...     "model": "all-MiniLM-L6-v2"
        ... }
        >>> provider = create_embedding_provider(config)
    """
    provider = config.get("provider", "local")  # Default to local
    model = config.get("model")

    if not model:
        raise ValueError("Embedding model not specified in config")

    if provider == "openrouter":
        api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        return OpenRouterEmbeddingProvider(model, api_key)

    elif provider == "local":
        return LocalEmbeddingProvider(model)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
