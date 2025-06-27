import os
import streamlit as st
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from chromadb.config import Settings

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# UI setup
st.set_page_config(page_title="📖 Intelligent Academic Search", layout="wide")
st.title("📖 Intelligent Academic Search with RAG")

uploaded_files = st.file_uploader("Upload your academic PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info("Saving and processing uploaded PDFs...")
    pdf_dir = "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join(pdf_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

    # Load documents
    docs = []
    for file in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        docs.extend(PyPDFLoader(file).load())

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ChromaDB setup
    chroma_db_dir = "chroma_db"
    chroma_settings = Settings(anonymized_telemetry=False)

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=chroma_db_dir,
        client_settings=chroma_settings
    )

    # LLM
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-base", temperature=0.7)

    # Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

    st.subheader("💬 Ask a question from the uploaded PDFs")
    query = st.text_input("Type your query here")

    if query:
        with st.spinner("Generating response..."):
            answer = qa_chain.run(query)
        st.success("Answer:")
        st.write(answer)
