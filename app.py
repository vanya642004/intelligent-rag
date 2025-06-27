import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import tempfile

st.set_page_config(page_title="📚 Academic RAG Assistant", layout="centered")
st.title("📄 Ask Your PDFs (RAG powered)")

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload PDF documents to get started.")
    st.stop()

all_chunks = []
for file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

# Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_chunks, embeddings)

# LLM Pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Query UI
query = st.text_input("Ask a question based on the uploaded PDFs:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
    st.success("Answer:")
    st.write(result)
