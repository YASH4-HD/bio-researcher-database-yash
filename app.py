import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.set_page_config(page_title="Bio-Researcher AI", layout="wide")

st.title("🧬 Molecular Biology Research Assistant")
st.write("Querying: **Lehninger Principles of Biochemistry**")

# 1. Load the Index we just created
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Path to your index folder
    index_path = "faiss_index"
    
    if os.path.exists(index_path):
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return db
    else:
        st.error("Index folder not found. Please run create_index.py first.")
        return None

db = load_vector_db()

# 2. Setup the Search logic
query = st.text_input("Enter your research question (e.g., 'What is the role of ATP synthase?'):")

if query and db:
    with st.spinner("Searching the textbook..."):
        # We use similarity_search directly from the FAISS object
        # This bypasses the need for the 'langchain.chains' module
        docs = db.similarity_search(query, k=4)
        
        st.subheader(f"Top results for: {query}")
        
        for i, doc in enumerate(docs):
            with st.expander(f"Source Snippet {i+1} (Page {doc.metadata.get('page', 'N/A')})"):
                st.write(doc.page_content)
                st.caption(f"Metadata: {doc.metadata}")
