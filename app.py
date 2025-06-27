import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryDocstore, SimpleVectorStore
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# UI setup
st.title("📚 Academic Assistant (FAISS-free RAG)")

# Upload PDFs
files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if not files:
    st.info("📂 Upload academic PDFs to get started")
    st.stop()

# Load & chunk docs
texts = []
for pdf in files:
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts.extend(splitter.split_documents(docs))

# Embed and store vectors in-memory
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = SimpleVectorStore.from_documents(texts, embeddings)

# Build retriever & RAG chain
retriever = store.as_retriever(search_kwargs={"k": 3})
qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=qa_pipe)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query
query = st.text_input("🔍 Ask your academic question")
if query:
    with st.spinner("🤖 Thinking with RAG..."):
        answer = qa.invoke({"question": query})["result"]
    st.markdown("### 📖 Answer")
    st.write(answer)
