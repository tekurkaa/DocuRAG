# 📄 DocuRAG

DocuRAG is an interactive **Retrieval-Augmented Generation (RAG)** tool that lets you process documents or URLs and ask questions about them.  
It uses **LangChain**, **FAISS**, and **OpenAI embeddings/LLMs** (or you could use any other model of your choice) to provide accurate answers with references to sources.

👉 **Live Demo:** [Click here to try the app](https://atv-docurag.streamlit.app/)


## 🚀 Features
- Upload documents (`.pdf`, `.txt`, `.docx`) or provide a URL.  
- Automatically splits and embeds text into a **FAISS vectorstore**.  
- Ask questions about the processed content.  
- Provides **answers with cited sources**.  
- Built with **Streamlit** for a simple web-based interface.  


## 🛠️ Tech Stack
- [Python 3.9+](https://www.python.org/)  
- [Streamlit](https://streamlit.io/) – frontend interface  
- [LangChain](https://www.langchain.com/) – document loading, chunking, retrieval  
- [FAISS](https://faiss.ai/) – vectorstore for embeddings  
- [OpenAI API](https://platform.openai.com/) – embeddings + LLM (can be swapped with any other model)  


## 📦 Installation

1. **Clone the repo**
    ```bash
    git clone https://github.com/your-username/DocuRAG.git
    cd DocuRAG
    ```

2. **Create and activate a virtual environment**
    ```
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Set up environment variables** \
    Create a `.env` file in the root folder
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```


## ▶️ Usage

Run the Streamlit app:
```
streamlit run main.py
```