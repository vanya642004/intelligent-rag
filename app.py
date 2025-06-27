import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from chromadb.config import Settings

# Set Hugging Face API token from Streamlit secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

st.set_page_config(page_title="AI Academic Assistant", layout="wide")
st.title("📖 Intelligent Academic Search with RAG")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_folder = "uploaded_pdfs"
    os.makedirs(pdf_folder, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(pdf_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    documents = []
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    st.info("📡 Creating vector database...")
    
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    chroma_settings = Settings(
        persist_directory="chroma_db",
        chroma_api_impl="local",
        anonymized_telemetry=False
    )

    vector_db = Chroma.from_documents(
        documents,
        embedding=embedding_function,
        persist_directory="chroma_db",
        client_settings=chroma_settings
    )

    # Use a valid lightweight model for HuggingFaceHub
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type="stuff"
    )

    st.subheader("💬 Ask any question based on the uploaded PDFs")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.success("Answer generated!")
        st.write("Answer:", response)
