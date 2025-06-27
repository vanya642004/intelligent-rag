import os
import streamlit as st
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Streamlit UI
st.set_page_config(page_title="AI Academic Assistant", layout="wide")
st.title("📖 Intelligent Academic Search with RAG")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_folder = "uploaded_pdfs"
    os.makedirs(pdf_folder, exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join(pdf_folder, file.name), "wb") as f:
            f.write(file.getbuffer())

    documents = []
    for path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    st.info("📡 Creating FAISS vector DB in memory...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever(), chain_type="stuff")

    st.subheader("💬 Ask any question based on the uploaded PDFs")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
        st.success("✅ Answer:")
        st.write(result)


