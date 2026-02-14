import streamlit as st
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ==========================================================
# 1️⃣ CONFIGURATION
# ==========================================================

PDF_PATH = "lehninger.pdf"

# ✅ CORRECT Dropbox Direct Link (dl=1 required)
DROPBOX_URL = "https://dl.dropboxusercontent.com/scl/fi/wzbf5ra623k6ex3pt98gc/lehninger.pdf?rlkey=fzauw5kna9tyyo2g336f8w5a0&dl=1"

# ==========================================================
# 2️⃣ DROPBOX SAFE DOWNLOADER
# ==========================================================

def extract_smart_visuals(page_num, mode="Smart Crop"):
    try:
        if not os.path.exists(PDF_PATH):
            return "file_not_found"

        doc = fitz.open(PDF_PATH)
        page = doc.load_page(int(page_num) - 1)

        # If Full Page View → just render
        if mode == "Full Page View":
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            return Image.open(io.BytesIO(pix.tobytes("png")))

        # Smart Crop (ONLY using drawings, no image_info)
        drawings = page.get_drawings()

        if drawings:
            rect = drawings[0]["rect"]
            for d in drawings[1:]:
                rect = rect | d["rect"]

            page.set_cropbox(rect + (-20, -20, 20, 20))

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return Image.open(io.BytesIO(pix.tobytes("png")))

    except Exception as e:
        return str(e)


# ==========================================================
# 3️⃣ VECTOR STORE LOADER
# ==========================================================

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name="lehninger-index",
        embedding=embeddings,
        pinecone_api_key=st.secrets["PINECONE_API_KEY"]
    )
    return vectorstore


# ==========================================================
# 4️⃣ VISUAL EXTRACTION FUNCTION
# ==========================================================

def extract_smart_visuals(page_num, mode="Smart Crop"):
    try:
        if not os.path.exists(PDF_PATH):
            return "file_not_found"

        doc = fitz.open(PDF_PATH)
        idx = int(page_num) - 1
        page = doc.load_page(idx)

        if mode == "Smart Crop":
            paths = page.get_drawings()
            images = page.get_image_info()

            bboxes = [p["rect"] for p in paths] + [i["bbox"] for i in images]

            if bboxes:
                v_rect = bboxes[0]
                for b in bboxes[1:]:
                    v_rect = v_rect | b
                page.set_cropbox(v_rect + (-15, -15, 15, 15))

        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        return Image.open(io.BytesIO(pix.tobytes("png")))

    except Exception as e:
        return str(e)


# ==========================================================
# 5️⃣ STREAMLIT PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="Bio-Researcher AI | Yashwant Nama",
    layout="wide",
    page_icon="🧬"
)

# ==========================================================
# 6️⃣ SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("👨‍🔬 Researcher Info")
    st.markdown("""
    **Yashwant Nama**  
    PhD Applicant | Molecular Biology  

    Project: Multimodal RAG for Metabolic Research
    """)
    st.divider()

    extraction_mode = st.radio(
        "Visual Extraction Mode:",
        ["Smart Crop", "Full Page View"]
    )

    st.divider()

    # Debug section (remove later)
    if st.checkbox("Show Debug Info"):
        st.write("PDF Exists:", os.path.exists(PDF_PATH))
        if os.path.exists(PDF_PATH):
            st.write("File Size (MB):", round(os.path.getsize(PDF_PATH)/1024**2, 2))


# ==========================================================
# 7️⃣ MAIN UI
# ==========================================================

st.title("🧬 Molecular Biology Research Assistant")
st.caption("AI-powered knowledge retrieval from Lehninger Principles of Biochemistry")

# Download PDF
pdf_ready = download_pdf()

if pdf_ready:

    docsearch = load_vectorstore()

    query = st.text_input(
        "Enter your research question:",
        placeholder="e.g. Describe transferases"
    )

    if query:
        with st.spinner("🔬 Searching metabolic database..."):

            results = docsearch.similarity_search(query, k=3)

            if not results:
                st.warning("No matches found in vector index.")

            for i, doc in enumerate(results):

                raw_page = doc.metadata.get("page", 0)
                clean_page = int(float(raw_page))

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown(f"### Result {i+1} | Page {clean_page}")
                    st.info(doc.page_content)

                with col2:
                    if st.button(f"🔍 Extract Visuals (P. {clean_page})", key=f"btn_{i}"):

                        with st.spinner("Extracting diagrams..."):
                            img = extract_smart_visuals(clean_page, extraction_mode)

                            if isinstance(img, Image.Image):
                                st.image(
                                    img,
                                    use_container_width=True,
                                    caption=f"Source: Page {clean_page}"
                                )
                            elif img == "file_not_found":
                                st.error("PDF file not found on server.")
                            else:
                                st.error(f"Extraction failed: {img}")

                st.divider()

else:
    st.error("❌ PDF could not be loaded. Check Dropbox link.")
