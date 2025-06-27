import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# UI setup
st.title("📚 Academic Assistant (RAG-powered)")
query = st.text_input("Enter your academic query")

# Load docs & embed
loader = TextLoader("docs/sample.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)

# Hugging Face Pipeline (FLAN-T5)
qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=qa_pipe)

# RAG chain
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Output
if query:
    with st.spinner("Searching..."):
        result = qa_chain.run(query)
        st.markdown("### 📖 Answer")
        st.write(result)
