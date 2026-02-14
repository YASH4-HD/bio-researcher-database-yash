import streamlit as st
import os
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Page Configuration ---
st.set_page_config(page_title="Bio-Researcher AI", layout="wide")

st.title("🧬 Molecular Biology Research Assistant")
st.write("Querying: **Lehninger Principles of Biochemistry (Cloud Edition)**")

# --- Constants & Secrets ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "lehninger-index"
PDF_PATH = "data/Lehninger_Biochemistry.pdf" # Ensure this path is correct in your repo

# --- Core Functions ---

@st.cache_resource
def load_pinecone():
    """Initializes connection to Pinecone Vector Database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore

def get_pdf_page_as_image(pdf_path, page_num, zoom=2):
    """Extracts a PDF page as an image for visual reference."""
    try:
        if not os.path.exists(pdf_path):
            return None
        
        doc = fitz.open(pdf_path)
        # Handle index offset: Lehninger PDF pages might differ from physical page numbers
        # Usually, metadata 'page' is 1-indexed, fitz is 0-indexed
        page = doc.load_page(int(page_num) - 1) 
        
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.error(f"Error rendering page: {e}")
        return None

# --- Main App Logic ---

docsearch = load_pinecone()

query = st.text_input("Enter your research question (e.g., 'Mechanism of ATP Synthase'):")

if query:
    with st.spinner("Searching Cloud Database..."):
        # Retrieve top 4 most relevant chunks
        results = docsearch.similarity_search(query, k=4)
        
        if not results:
            st.warning("No relevant matches found.")
        
        for i, doc in enumerate(results):
            page_val = doc.metadata.get('page', 'N/A')
            
            # Create a container for each result
            with st.container():
                st.markdown(f"### Result {i+1}")
                
                col1, col2 = st.columns([1, 1]) # Split view: Text on left, Image on right
                
                with col1:
                    st.info(f"**Extracted Text (Page {page_val})**")
                    st.write(doc.page_content)
                
                with col2:
                    if page_val != 'N/A':
                        if st.button(f"📷 View Diagrams on Page {page_val}", key=f"btn_{i}"):
                            with st.spinner("Rendering diagram..."):
                                page_img = get_pdf_page_as_image(PDF_PATH, page_val)
                                if page_img:
                                    st.image(page_img, use_container_width=True, caption=f"Lehninger Page {page_val}")
                                else:
                                    st.error("PDF file not found in repository path.")
                    else:
                        st.warning("No page metadata available for this chunk.")
                
                st.divider()

# --- Requirements Check ---
# Ensure your requirements.txt contains:
# streamlit
# langchain-huggingface
# langchain-pinecone
# pymupdf
# pillow
