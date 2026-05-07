"""Default document loader adapters using LangChain loaders.

Imports from heavy external packages are performed lazily inside methods
so unit tests can import these modules without requiring the heavy
dependencies when using fakes.
"""
import os
import tempfile
from typing import Any, List, Optional

from .interfaces import DocumentLoader
from .errors import DocumentLoadError
from .config import MAX_UPLOAD_BYTES, TEMP_DIR


class LangchainDocumentLoader(DocumentLoader):
    """A concrete DocumentLoader that wraps LangChain community loaders.

    This implementation performs an early file-size check so oversized
    uploads are rejected before attempting to import heavy libraries.
    """

    def load(self, url: Optional[str] = None, uploaded_file: Any = None) -> List[Any]:
        docs = []

        # URL loading (simple, with early failure)
        if url:
            try:
                from langchain_community.document_loaders import UnstructuredURLLoader

                loader = UnstructuredURLLoader(urls=[url])
                docs.extend(loader.load())
            except Exception as exc:
                raise DocumentLoadError(f"Failed to load URL {url}: {exc}") from exc

        # Uploaded file handling: check size first, then parse with LangChain
        if uploaded_file:
            os.makedirs(TEMP_DIR, exist_ok=True)

            # obtain bytes for size check without importing heavy loaders
            try:
                buf = uploaded_file.getbuffer()
                data = bytes(buf)
            except Exception:
                # fallback to stream read
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                data = uploaded_file.read()
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass

            size = len(data) if data is not None else 0
            if size > MAX_UPLOAD_BYTES:
                raise DocumentLoadError(
                    f"Uploaded file exceeds maximum size ({MAX_UPLOAD_BYTES} bytes)"
                )

            # write to a safe temporary file (unique name)
            suffix = os.path.splitext(uploaded_file.name)[1] if hasattr(uploaded_file, "name") else ""
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                # pick loader based on extension
                lower_name = (uploaded_file.name or "").lower()
                if lower_name.endswith(".pdf"):
                    from langchain_community.document_loaders import PyPDFLoader

                    loader = PyPDFLoader(tmp_path)
                elif lower_name.endswith(".txt"):
                    from langchain_community.document_loaders import TextLoader

                    loader = TextLoader(tmp_path)
                elif lower_name.endswith(".docx"):
                    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

                    loader = UnstructuredWordDocumentLoader(tmp_path)
                else:
                    raise DocumentLoadError("Unsupported file format")

                try:
                    file_docs = loader.load()
                except Exception as exc:
                    raise DocumentLoadError(f"Failed to parse uploaded file: {exc}") from exc

                for doc in file_docs:
                    # some loader implementations may not expose metadata dicts
                    try:
                        doc.metadata["source"] = uploaded_file.name
                    except Exception:
                        pass
                docs.extend(file_docs)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        return docs
