import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import tempfile

st.set_page_config(page_title="PDF Q&A")
st.title("📄 Ask a question from your PDF")

query = st.text_input("Enter your question")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # 🔥 KEY FIX HERE
    response = qa.run(query)

    st.success(response)
