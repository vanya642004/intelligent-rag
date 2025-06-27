import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

st.set_page_config(page_title="AI Academic Assistant", layout="wide")
st.title("📖 Intelligent Academic Search with RAG")

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

    # ✅ Split the documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    st.info("📡 Creating vector database...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory="chroma_db"
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
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
