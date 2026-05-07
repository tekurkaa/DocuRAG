# 📄 DocuRAG

DocuRAG is an interactive Retrieval-Augmented Generation (RAG) application
that lets you process documents or URLs and ask natural-language questions
about their content. It demonstrates a typical RAG workflow: ingest → split
→ embed → index → retrieve → generate.

👉 **Live Demo:** [Click here to try the app](https://atv-docurag.streamlit.app/)


## Overview

- Upload documents (`.pdf`, `.txt`, `.docx`) or provide a URL to ingest.
- Split documents into overlapping chunks for retrieval.
- Create embeddings and store vectors in a local FAISS index.
- Query the index and generate answers with an LLM (Gemini/other).
- Simple Streamlit UI for demos; the core pipeline is reusable.

## Project layout

- `DocuRAG.py` — Streamlit user interface (thin wrapper).
- `core/` — backend modules (loader, embedding wrapper, retriever, pipeline, config, errors).
- `tests/` — unit tests covering core behaviors.
- `faiss_store_gemini/` — local FAISS index directory (ignored by git).

## Tech stack

- Python 3.9+
- Streamlit for demo UI
- LangChain and LangChain community loaders
- FAISS (vector store)
- Google Gemini (optional) or any compatible embedding/LLM provider

## Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/DocuRAG.git
cd DocuRAG
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the repository root with at least your API key(s):

```
GEMINI_API_KEY=your_gemini_api_key
# or use GOOGLE_API_KEY if applicable
```

Other runtime configuration options are in `core/config.py`:

- `INDEX_PATH` — path for the FAISS index (default: `faiss_store_gemini`).
- `MAX_UPLOAD_BYTES` — maximum allowed upload size for files.
- `ALLOW_DANGEROUS_DESERIALIZATION` — when `True`, allows deserializing
  certain index files that require unpickling. This can execute arbitrary
  code if the index file is from an untrusted source; set to `False` in
  production or when loading indexes from unknown origins.

If you want to rebuild the index instead of loading an existing one,
delete the `faiss_store_gemini/` directory and re-process your documents.

## Usage

Run the Streamlit demo:

```bash
streamlit run DocuRAG.py
```

Follow the UI to paste a URL or upload a file, process the data to build
an index, and then ask questions about the processed content.

## Tests

Run unit tests with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests use lightweight fakes so they run without heavy third-party LLM
or FAISS packages.

## Security & privacy

- Never commit your `.env` or API keys to source control. `.gitignore` is
  configured to exclude `.env`, virtual environments, and the FAISS index
  folder.
- Uploaded documents are stored temporarily during processing and removed
  as soon as parsing completes. Treat any deserialization options with
  care — avoid enabling dangerous deserialization for untrusted index
  files.

## Extending & Productionizing

This project is suitable as a demo or prototype. To move to production
consider:

- Extracting a service layer (FastAPI) with authentication and rate limits.
- Using a managed vector database (e.g., Milvus, Chroma Cloud) for scale.
- Adding monitoring and metrics for retrieval quality and index health.
- Encrypting on-disk data if storing sensitive documents.
