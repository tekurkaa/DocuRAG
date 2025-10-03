# --------------------
#       IMPORTS
# --------------------
import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from rag_pipeline import RAGPipeline


# --------------------
#       SETUP
# --------------------
load_dotenv()   # load API keys from .env
st.set_page_config(page_title="Research Tool", page_icon="🖥️", layout="wide")   # page name and icon
st.title("🖥️ Research Tool")    # page title
st.markdown("Paste any article URL in the sidebar or upload a file, process them, then ask questions below.")


# --------------------
#       SIDEBAR
# --------------------
st.sidebar.header("Paste URL or upload document")
url = st.sidebar.text_input("🔗 Enter a URL")
uploaded_file = st.sidebar.file_uploader("📂 Upload a document", 
                                         type=["pdf", "txt", "docx"],
                                         accept_multiple_files=False)
process_clicked = st.sidebar.button("🔍 Process data")


# ----------------------------------------
#       INITIALIZE LLM + EMBEDDINGS
# ----------------------------------------
llm = OpenAI(temperature = 0.7, max_tokens=1000)
embeddings = OpenAIEmbeddings()

pipeline = RAGPipeline(llm, embeddings)

main_placeholder = st.empty()


# -------------------------------
#       PROCESS URL + FILE
# -------------------------------
if process_clicked:
    if not url and not uploaded_file:
        st.sidebar.warning("Please provide either a URL or a file.")
    else:
        try:
            # load data
            main_placeholder.info("⏳ Loading documents...")
            docs = pipeline.load_documents(url, uploaded_file)
            time.sleep(1)

            if not docs:
                st.error("❌ Failed to fetch or parse content from the given URL or file.")
            else:
                # split data
                main_placeholder.info("✂️ Splitting text into chunks...")
                split_docs = pipeline.split_documents(docs)
                time.sleep(1)

                if not split_docs:
                    st.error("❌ No text could be extracted from the URL or file.")
                else:
                    # create embeddings and save it to FAISS index
                    main_placeholder.info("⚡ Creating embeddings...")
                    pipeline.index_documents(split_docs)
                    time.sleep(1)

                    main_placeholder.success("✅ Processing complete! You can now ask questions.")

        except Exception as e:
            st.error(f"⚠️ Error fetching or processing: {str(e)}")


# -------------------------------
#       QUESTION INPUT UI
# -------------------------------
st.markdown("---")
st.subheader("💬 Ask a Question")

with st.form(key="qa_form", clear_on_submit=False):
    col1, col2 = st.columns([18, 1])
    with col1:
        query = st.text_input(
            "Ask your question here:", 
            label_visibility="collapsed", 
            placeholder="Type your question..."
        )
    with col2:
        send = st.form_submit_button("➤")


# -------------------------------
#       RETRIEVE ANSWER
# -------------------------------
if send:
    if not query.strip():
        st.warning("⚠️ Please enter a valid question before sending.")
    elif not os.path.exists(pipeline.vectorstore_path):
        st.error("❌ No processed data found. Please process URLs first.")
    else:
        try:
            # load FAISS index
            pipeline.load_index()

            # run query
            with st.spinner("🤔 Thinking..."):
                result = pipeline.query(query)

            # validate result
            if not result or not result.get("answer", "").strip():
                st.warning("⚠️ No answer could be generated for this question.")
            else:
                st.markdown("### 📌 Answer")
                st.write(result["answer"])

                st.markdown("---")

                # show sources if available
                sources = result.get("sources", "")
                if sources:
                    st.markdown("### 🌐 Sources")
                    for source in sources.split("\n"):
                        if source.strip():
                            st.write(f"- {source}")

        except Exception as e:
            st.error(f"⚠️ An error occurred while retrieving the answer: {str(e)}")
