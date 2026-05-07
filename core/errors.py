"""Custom exception classes for DocuRAG.

Centralize error types so callers (UI/service/tests) can handle
different failure modes explicitly.
"""

class DocuRAGError(Exception):
    """Base class for all DocuRAG-specific exceptions."""


class DocumentLoadError(DocuRAGError):
    """Raised when a document cannot be loaded or parsed."""


class EmbeddingError(DocuRAGError):
    """Raised when embedding generation fails or returns invalid results."""


class IndexingError(DocuRAGError):
    """Raised when an index build or save operation fails."""


class RetrievalError(DocuRAGError):
    """Raised when an index cannot be loaded or a retrieval query fails."""


class ConfigError(DocuRAGError):
    """Raised on invalid configuration or environment problems."""

__all__ = [
    "DocuRAGError",
    "DocumentLoadError",
    "EmbeddingError",
    "IndexingError",
    "RetrievalError",
    "ConfigError",
]
