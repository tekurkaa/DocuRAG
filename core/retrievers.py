"""Retriever adapter implementations.

Provides a FAISS-backed Retriever that implements the `Retriever`
interface. Heavy imports are performed lazily inside methods.
"""
from typing import Any, List, Optional
import os
import time
import shutil

from .interfaces import Retriever
from .errors import IndexingError, RetrievalError
from .config import ALLOW_DANGEROUS_DESERIALIZATION


class FAISSRetriever(Retriever):
    def __init__(self, vectorstore_path: str = "faiss_store_gemini"):
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None

    def create_index(self, split_docs: List[Any], embeddings: Any, vectorstore_path: Optional[str] = None):
        """Create and persist a FAISS index from document chunks.

        This writes the index to a temporary directory first and then
        replaces the target directory to reduce the risk of leaving a
        partially-written index behind.
        """
        try:
            from langchain_community.vectorstores import FAISS

            path = vectorstore_path or self.vectorstore_path
            tmp_path = f"{path}_tmp_{int(time.time())}"

            vs = FAISS.from_documents(split_docs, embeddings)
            vs.save_local(tmp_path)

            # atomically replace existing index directory with the new one
            if os.path.exists(path):
                shutil.rmtree(path)
            os.replace(tmp_path, path)

            # load final vectorstore reference
            self.vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=ALLOW_DANGEROUS_DESERIALIZATION)
        except Exception as exc:
            raise IndexingError(f"Failed to create index: {exc}") from exc

    def load_index(self, embeddings: Any, vectorstore_path: Optional[str] = None):
        """Load a persisted FAISS index and set `self.vectorstore`."""
        try:
            from langchain_community.vectorstores import FAISS

            path = vectorstore_path or self.vectorstore_path
            self.vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=ALLOW_DANGEROUS_DESERIALIZATION)
            return self.vectorstore
        except Exception as exc:
            raise RetrievalError(f"Failed to load index from {path}: {exc}") from exc

    def run_qa(self, llm: Any, query: str) -> dict:
        """Run a RetrievalQA chain using the internal vectorstore."""
        if not self.vectorstore:
            raise RetrievalError("Vectorstore not loaded. Call load_index() first.")

        try:
            from langchain_classic.chains import RetrievalQAWithSourcesChain

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=self.vectorstore.as_retriever())
            return chain({"question": query}, return_only_outputs=True)
        except Exception as exc:
            raise RetrievalError(f"Retrieval QA failed: {exc}") from exc
