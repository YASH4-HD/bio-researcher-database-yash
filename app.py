import streamlit as st
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import urllib.request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 1. Configuration & Dropbox Setup ---
PDF_PATH = "lehninger.pdf"
# Your Dropbox link converted to a Direct Download link (dl=1)
DROPBOX_URL = "https://www.dropbox.com/scl/fi/wzbf5ra623k6ex3pt98gc/lehninger.pdf?rlkey=fzauw5kna9tyyo2g336f8w5a0&st=yhsb62iw&dl=1"

@st.cache_data(show_spinner=False)
def download_pdf():
    """Downloads the PDF from Dropbox to the Streamlit server if not present."""
    if not os.path.exists(PDF_PATH):
        try:
            # Add headers to mimic a browser request (prevents 403 Forbidden)
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            with st.spinner("📥 Downloading Lehninger Database (Initial setup)... Please wait."):
                urllib.request.urlretrieve(DROPBOX_URL, PDF_PATH)
            st.success("✅ Database linked successfully!")
        except Exception as e:
            st.error(f"❌ Download failed: {e}")

# Trigger the download
download_pdf()

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="Bio-Researcher AI | Yashwant Nama", 
    layout="wide",
    page_icon="🧬"
)

# --- 3. Logic Functions ---

@st.cache_resource
def load_vectorstore():
    """Connects to the Pinecone cloud vector database."""
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    INDEX_NAME = "lehninger-index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore

def extract_smart_visuals(page_num, mode="Smart Crop"):
    """
    Extracts visuals from the PDF. 
    'Smart Crop' uses bounding box logic to isolate diagrams.
    """
    try:
        if not os.path.exists(PDF_PATH):
            return "file_not_found"
        
        doc = fitz.open(PDF_PATH)
        # Fix: Convert float (e.g., 1024.0) to int for PyMuPDF
        idx = int(float(page_num)) - 1
        page = doc.load_page(idx)
        
        if mode == "Smart Crop":
            # Detect drawings (vectors) and images (rasters)
            paths = page.get_drawings()
            images = page.get_image_info()
            bboxes = [p["rect"] for p in paths] + [i["bbox"] for i in images]
            
            if bboxes:
                # Calculate the union area of all detected visual elements
                v_rect = bboxes[0]
                for b in bboxes[1:]:
                    v_rect = v_rect | b
                # Apply crop with padding
                page.set_cropbox(v_rect + (-15, -15, 15, 15))
            
        # Render the page/crop to an image
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        return Image.open(io.BytesIO(pix.tobytes("png")))
        
    except Exception as e:
        return str(e)

# --- 4. Sidebar & Profile ---
with st.sidebar:
    st.title("👨‍🔬 Researcher Info")
    st.markdown("""
    **Yashwant Nama**  
    *PhD Applicant | Molecular Biology*
    
    **Project:** Multimodal RAG for Metabolic Research.
    """)
    st.divider()
    extraction_mode = st.radio(
        "Visual Extraction Mode:",
        ["Smart Crop (Focus on Diagrams)", "Full Page View"]
    )
    st.divider()
    if st.checkbox("Show Server Files (Debug)"):
        if os.path.exists(PDF_PATH):
            st.write(f"✅ {PDF_PATH} exists ({os.path.getsize(PDF_PATH)//1024**2} MB)")
        else:
            st.write("❌ PDF missing.")
        st.write("Files in root:", os.listdir("."))

# --- 5. Main UI Flow ---
st.title("🧬 Molecular Biology Research Assistant")
st.caption("Quantitative Analysis of Lehninger Principles of Biochemistry")

if os.path.exists(PDF_PATH):
    docsearch = load_vectorstore()
    query = st.text_input("Enter your research question:", placeholder="e.g. Describe the regulation of the Citric Acid Cycle")

    if query:
        with st.spinner("Analyzing metabolic pathways..."):
            results = docsearch.similarity_search(query, k=3)
            
            if not results:
                st.warning("No matches found in the vector index.")
            
            for i, doc in enumerate(results):
                # Clean up the .0 metadata issue
                raw_page = doc.metadata.get('page', 0)
                clean_page = int(float(raw_page))
                
                with st.container():
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"### Result {i+1} | Page {clean_page}")
                        st.info(doc.page_content)
                    
                    with col2:
                        if st.button(f"🔍 Extract Visuals (P. {clean_page})", key=f"btn_{i}"):
                            with st.spinner("Isolating diagrams..."):
                                img = extract_smart_visuals(clean_page, mode=extraction_mode)
                                if isinstance(img, Image.Image):
                                    st.image(img, use_container_width=True, caption=f"Source: Lehninger Page {clean_page}")
                                elif img == "file_not_found":
                                    st.error("PDF file missing from server storage.")
                                else:
                                    st.error(f"Extraction failed: {img}")
                    st.divider()
else:
    st.info("🔄 System is initializing. Please wait for the database download to complete...")
