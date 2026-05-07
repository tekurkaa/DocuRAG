"""Pluggable component interfaces for DocuRAG.

Define abstract interfaces for document loaders, embedders, and retrievers
so the pipeline can accept interchangeable implementations and be tested
with lightweight fakes.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, url: Optional[str] = None, uploaded_file: Any = None) -> List[Any]:
        """Load documents from a URL or uploaded file and return a list of
        document-like objects.
        """
        raise NotImplementedError


class Embedder(ABC):
    @abstractmethod
    def embed_query(self, text: str):
        """Return an embedding vector for a single query string."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, docs: List[Any]):
        """Return embedding vectors for a list of documents or strings."""
        raise NotImplementedError


class Retriever(ABC):
    @abstractmethod
    def create_index(self, split_docs: List[Any], embeddings: Any, vectorstore_path: str):
        """Create and persist a vector index from document chunks."""
        raise NotImplementedError

    @abstractmethod
    def load_index(self, embeddings: Any, vectorstore_path: str):
        """Load a persisted index and set internal state for retrieval."""
        raise NotImplementedError

    @abstractmethod
    def run_qa(self, llm: Any, query: str) -> dict:
        """Run a retrieval-augmented QA using the provided LLM and query.

        Returns a dict with at least 'answer' and optionally 'sources'.
        """
        raise NotImplementedError
