import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

st.set_page_config(page_title="📖 Intelligent Academic Search with RAG")
st.title("📖 Intelligent Academic Search with RAG")

uploaded_files = st.file_uploader("📂 Upload your academic PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        all_texts.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_texts, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = st.text_input("💡 Ask your academic question")
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
        st.success("📖 Answer:")
        st.write(result)
