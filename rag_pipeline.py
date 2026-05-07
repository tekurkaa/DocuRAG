"""Legacy wrapper for the core pipeline.

This module provides a thin compatibility wrapper that re-exports
the `RAGPipeline` from `core.pipeline`. Existing code that imports
from `rag_pipeline` will continue to work while the reusable core
implementation lives in `core.pipeline`.
"""

from core.pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
    