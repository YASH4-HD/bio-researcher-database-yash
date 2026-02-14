import streamlit as st
import os
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Bio-Researcher AI | Yashwant Nama", 
    layout="wide",
    page_icon="🧬"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stExpander { background-color: white !important; border-radius: 10px; }
    </style>
    """, unsafe_allow_index=True)

st.title("🧬 Molecular Biology Research Assistant")
st.caption("Advanced RAG System: Querying Lehninger Principles of Biochemistry")

# --- 2. Configuration & Secrets ---
# Update 'lehninger.pdf' to match the exact filename in your GitHub
PDF_PATH = "lehninger.pdf" 
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "lehninger-index"

# --- 3. Logic Functions ---

@st.cache_resource
def load_vectorstore():
    """Connects to the Pinecone cloud vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore

def extract_smart_visuals(pdf_path, page_num):
    """
    Tries to extract specific image objects. 
    If none found, renders the full page as a fallback.
    """
    try:
        if not os.path.exists(pdf_path):
            return "file_not_found"
        
        doc = fitz.open(pdf_path)
        # Convert float metadata (e.g., 1024.0) to int for fitz
        idx = int(float(page_num)) - 1
        page = doc.load_page(idx)
        
        image_list = page.get_images(full=True)
        extracted_imgs = []

        if image_list:
            # Method A: Extract high-res image objects
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                extracted_imgs.append(Image.open(io.BytesIO(image_bytes)))
            return extracted_imgs
        else:
            # Method B: Render whole page if it's vector-based art
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return [img]
    except Exception as e:
        return str(e)

# --- 4. Main UI Flow ---

docsearch = load_vectorstore()

# Sidebar for researcher profile/info
with st.sidebar:
    st.header("Researcher Profile")
    st.info("**Name:** Yashwant Nama\n\n**Focus:** Computational Biology")
    st.write("---")
    st.write("**System Status:**")
    if os.path.exists(PDF_PATH):
        st.success("✅ PDF Database Linked")
    else:
        st.error("❌ PDF Not Found in Repo")
        st.caption(f"Looking for: {PDF_PATH}")

query = st.text_input("Enter your biological query:", placeholder="e.g. Describe the role of Carnitine in fatty acid oxidation")

if query:
    with st.spinner("Analyzing metabolic pathways..."):
        # Retrieve results
        results = docsearch.similarity_search(query, k=4)
        
        if not results:
            st.warning("No matches found in the vector index.")
        
        for i, doc in enumerate(results):
            # Clean up page number from metadata
            raw_page = doc.metadata.get('page', 'N/A')
            
            with st.expander(f"Result {i+1} | Source: Lehninger Page {raw_page}"):
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown("**Contextual Text Snippet:**")
                    st.write(doc.page_content)
                
                with col2:
                    if raw_page != 'N/A':
                        if st.button(f"🖼️ Extract Visuals (P. {raw_page})", key=f"btn_{i}"):
                            visuals = extract_smart_visuals(PDF_PATH, raw_page)
                            
                            if visuals == "file_not_found":
                                st.error("PDF file missing from server.")
                            elif isinstance(visuals, list):
                                for img in visuals:
                                    st.image(img, use_container_width=True)
                            else:
                                st.error(f"Error: {visuals}")
                    else:
                        st.info("No page metadata found.")

# --- 5. Debug Mode (Optional) ---
if st.checkbox("Show Debug Info"):
    st.write("Current Directory Files:", os.listdir("."))
