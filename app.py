import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import fitz  # PyMuPDF
from PIL import Image
st.set_page_config(page_title="Bio-Researcher AI", layout="wide")

st.title("🧬 Molecular Biology Research Assistant")
st.write("Querying: **Lehninger Principles of Biochemistry (Cloud Edition)**")

# 1. Setup API Key (On Streamlit Cloud, we will use 'Secrets')
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
index_name = "lehninger-index"

# 2. Connect to the Cloud Brain
@st.cache_resource
def load_pinecone():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings, 
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore

docsearch = load_pinecone()

# 3. Search UI
query = st.text_input("Enter your research question:")

if query:
    with st.spinner("Searching Cloud Database..."):
        results = docsearch.similarity_search(query, k=4)
        for i, doc in enumerate(results):
            with st.expander(f"Source {i+1} (Page {doc.metadata.get('page', 'N/A')})"):
                st.write(doc.page_content)
