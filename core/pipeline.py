"""High-level RAG pipeline built from pluggable components.

The pipeline delegates document loading and retrieval to separate
components that implement the interfaces defined in
`core.interfaces`. This keeps the pipeline testable and reusable by
different frontends.
"""

from typing import Any, List, Optional

from .interfaces import DocumentLoader, Retriever
from .errors import EmbeddingError


class RAGPipeline:
    def __init__(
        self,
        llm: Any = None,
        embeddings: Any = None,
        vectorstore_path: str = "faiss_store_gemini",
        loader: Optional[DocumentLoader] = None,
        retriever: Optional[Retriever] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        """Initialize the pipeline with optional pluggable components.

        For backward compatibility, `embeddings` can be the original
        embeddings object used by FAISS. If `loader` or `retriever` are
        omitted, defaults from `core.loaders` and `core.retrievers` are
        used lazily.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # lazily import default implementations only when needed
        if loader is None:
            from .loaders import LangchainDocumentLoader

            loader = LangchainDocumentLoader()
        if retriever is None:
            from .retrievers import FAISSRetriever

            retriever = FAISSRetriever(vectorstore_path)

        self.loader: DocumentLoader = loader
        self.retriever: Retriever = retriever
        self.vectorstore = None

    def load_documents(self, url: Optional[str] = None, uploaded_file: Any = None) -> List[Any]:
        return self.loader.load(url=url, uploaded_file=uploaded_file)

    def split_documents(self, docs: List[Any]) -> List[Any]:
        # lazy import to keep module import lightweight
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(docs)

    def index_documents(self, split_docs: List[Any]):
        if self.embeddings is None:
            raise EmbeddingError("Embeddings are required to create an index.")
        # delegate to retriever which will raise IndexingError on failure
        self.retriever.create_index(split_docs, self.embeddings, self.vectorstore_path)
        # keep a reference for compatibility
        self.vectorstore = getattr(self.retriever, "vectorstore", None)

    def load_index(self):
        if self.embeddings is None:
            raise EmbeddingError("Embeddings are required to load an index.")
        self.vectorstore = self.retriever.load_index(self.embeddings, self.vectorstore_path)
        return self.vectorstore

    def query(self, user_query: str) -> dict:
        return self.retriever.run_qa(self.llm, user_query)

