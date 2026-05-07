"""Configuration constants for DocuRAG.

Keep simple runtime-config values here so code can reference and tests
can override them if necessary.
"""
from typing import Final

# Maximum upload size accepted (bytes). Default 10 MiB.
MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024

# Default FAISS index folder
INDEX_PATH: Final[str] = "faiss_store_gemini"

# Whether to allow dangerous deserialization when loading FAISS indexes.
# Default to False for safety; set to True only for trusted local files.
ALLOW_DANGEROUS_DESERIALIZATION: Final[bool] = True

# URL loader retry/backoff settings
URL_LOAD_RETRIES: Final[int] = 2
URL_LOAD_TIMEOUT: Final[int] = 10

# Temporary directory used for file uploads and intermediate artifacts
TEMP_DIR: Final[str] = "temp"
