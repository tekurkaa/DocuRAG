# Imports
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline for loading,
    chunking, embedding, indexing, and querying documents.
    """

    def __init__(self, llm, embeddings, vectorstore_path="faiss_store_openai"):
        """
        Initialize the RAG pipeline.

        Args:
            llm: Language model instance (e.g., OpenAI).
            embeddings: Embedding model instance.
            vectorstore_path (str): Path to save/load FAISS index.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None


    def load_documents(self, url=None, uploaded_file=None):
        """
        Load documents from a URL or uploaded file.

        Args:
            url (str, optional): Webpage URL to load content from.
            uploaded_file: Uploaded file object (pdf, txt, docx).

        Returns:
            list: List of loaded Document objects.
        """
        docs = []

        # From URL
        if url:
            loader = UnstructuredURLLoader(urls=[url])
            docs.extend(loader.load())

        # From File
        if uploaded_file:
            uploaded_file_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Choose loader
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(uploaded_file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(uploaded_file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(uploaded_file_path)
            else:
                raise ValueError("Unsupported file format")

            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata["source"] = uploaded_file.name
            docs.extend(file_docs)

            # cleanup
            os.remove(uploaded_file_path)

        return docs


    def split_documents(self, docs):
        """
        Split documents into smaller overlapping text chunks.

        Args:
            docs (list): List of Document objects.

        Returns:
            list: List of Document chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=100
        )
        return splitter.split_documents(docs)


    def index_documents(self, split_docs):
        """
        Create FAISS embeddings index from documents and save locally.

        Args:
            split_docs (list): List of Document chunks.
        """
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        self.vectorstore.save_local(self.vectorstore_path)


    def load_index(self):
        """
        Load an existing FAISS index from disk.

        Returns:
            FAISS: Loaded FAISS vectorstore.
        """
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore


    def query(self, user_query):
        """
        Query the FAISS index with a natural language question.

        Args:
            user_query (str): Input question.

        Returns:
            dict: Dictionary with 'answer' and 'sources'.
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not loaded. Call load_index() first.")

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever()
        )
        return chain({"question": user_query}, return_only_outputs=True)
    