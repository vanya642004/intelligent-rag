import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from huggingface_hub import login

# Load environment variables
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=huggingface_token)

st.title("📚 Intelligent Academic Search")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    pdf = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf.pages:
        raw_text += page.extract_text()

    # Chunk text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
    vector_db = FAISS.from_texts(chunks, embeddings)

    # LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    # RetrievalQA
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Query box
    query = st.text_input("Ask something from the document")
    if st.button("Search"):
        with st.spinner("🔍 Searching..."):
            response = qa_chain.invoke({"query": query})
            st.write("🧠 Answer:", response["result"])
