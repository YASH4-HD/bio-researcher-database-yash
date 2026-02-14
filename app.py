import streamlit as st
import os
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import urllib.request

PDF_PATH = "lehninger.pdf"

@st.cache_data
def download_pdf():
    if not os.path.exists(PDF_PATH):
        # This is the converted direct link from your ID
        file_id = "1QvDN1bAnWYg2DC5ZyZOBbKdYvoNNFqvM"
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            with st.spinner("Downloading Lehninger PDF (this may take a minute)..."):
                # Using a User-Agent header helps prevent Google from blocking the script
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(direct_url, PDF_PATH)
            st.success("Database downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {e}")

download_pdf()

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Bio-Researcher AI | Yashwant Nama", 
    layout="wide",
    page_icon="🧬"
)

# --- 2. Configuration & Secrets ---
# Ensure this matches your file in GitHub exactly
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

def extract_smart_visuals(pdf_path, page_num, mode="Smart Crop"):
    """
    Extracts visuals from PDF. 
    Mode 'Smart Crop' isolates diagrams. 'Full Page' shows everything.
    """
    try:
        if not os.path.exists(pdf_path):
            return "file_not_found"
        
        doc = fitz.open(pdf_path)
        idx = int(float(page_num)) - 1
        page = doc.load_page(idx)
        
        if mode == "Smart Crop":
            # Identify drawing and image areas
            paths = page.get_drawings()
            images = page.get_image_info()
            bboxes = [p["rect"] for p in paths] + [i["bbox"] for i in images]
            
            if bboxes:
                # Calculate the union of all visual elements
                v_rect = bboxes[0]
                for b in bboxes[1:]:
                    v_rect = v_rect | b
                # Set cropbox with a bit of padding
                page.set_cropbox(v_rect + (-15, -15, 15, 15))
            
        # Render high-resolution image
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
        st.write(os.listdir("."))

# --- 5. Main UI Flow ---
st.title("🧬 Molecular Biology Research Assistant")
st.write(f"Active Database: `{INDEX_NAME}`")

docsearch = load_vectorstore()
query = st.text_input("Enter your research question:", placeholder="e.g. How does Glucose-6-Phosphate regulate glycolysis?")

if query:
    with st.spinner("Querying vector space..."):
        results = docsearch.similarity_search(query, k=3)
        
        if not results:
            st.warning("No matches found.")
        
        for i, doc in enumerate(results):
            # Resolve the .0 metadata issue
            raw_page = doc.metadata.get('page', 0)
            clean_page = int(float(raw_page))
            
            with st.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"### Result {i+1} (Page {clean_page})")
                    st.info(doc.page_content)
                
                with col2:
                    if st.button(f"🔍 Extract Visuals from P. {clean_page}", key=f"btn_{i}"):
                        with st.spinner("Processing PDF..."):
                            img = extract_smart_visuals(PDF_PATH, clean_page, mode=extraction_mode)
                            if isinstance(img, Image.Image):
                                st.image(img, use_container_width=True, caption=f"Visuals extracted from Lehninger, Page {clean_page}")
                            elif img == "file_not_found":
                                st.error(f"Error: '{PDF_PATH}' not found in root directory.")
                            else:
                                st.error(f"Extraction failed: {img}")
                st.divider()

