import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

st.title("🧠 In-Memory RAG Academic Assistant")

files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if not files:
    st.info("Upload one or more PDFs to get started.")
    st.stop()

chunks = []
for pdf in files:
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks.extend(splitter.split_documents(docs))

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = InMemoryVectorStore.from_documents(chunks, embedder)

qa_pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=qa_pipe)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

query = st.text_input("Enter your academic question")
if query:
    with st.spinner("Generating answer…"):
        output = qa(query)
    st.markdown("### 📖 Answer")
    st.write(output)
