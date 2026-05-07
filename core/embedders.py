"""Embedder adapter/wrapper utilities.

Provides a simple wrapper for embedding objects to match the `Embedder`
interface. The wrapper delegates to the underlying embeddings object
and validates results, converting errors to `EmbeddingError`.
"""
from typing import Any, List

from .interfaces import Embedder
from .errors import EmbeddingError


class WrapperEmbedder(Embedder):
    def __init__(self, embeddings: Any):
        self._embeddings = embeddings

    def embed_query(self, text: str):
        try:
            if hasattr(self._embeddings, "embed_query"):
                vec = self._embeddings.embed_query(text)
            elif hasattr(self._embeddings, "embed_documents"):
                vecs = self._embeddings.embed_documents([text])
                vec = vecs[0] if vecs else None
            else:
                raise EmbeddingError("Underlying embeddings object has no embed methods")

            if vec is None:
                raise EmbeddingError("Received empty embedding for query")
            return vec
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding query failed: {exc}") from exc

    def embed_documents(self, docs: List[Any]):
        texts: List[str] = []
        for d in docs:
            if hasattr(d, "page_content"):
                texts.append(d.page_content)
            elif isinstance(d, str):
                texts.append(d)
            else:
                texts.append(str(d))

        try:
            if hasattr(self._embeddings, "embed_documents"):
                vectors = self._embeddings.embed_documents(texts)
            elif hasattr(self._embeddings, "embed_query"):
                vectors = [self._embeddings.embed_query(t) for t in texts]
            else:
                raise EmbeddingError("Underlying embeddings object has no embed methods")

            if vectors is None or len(vectors) != len(texts):
                raise EmbeddingError(
                    f"Embedding returned {len(vectors) if vectors is not None else 'None'} vectors for {len(texts)} texts"
                )

            return vectors
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding documents failed: {exc}") from exc
