import streamlit as st
import os
import fitz
from PIL import Image
import io
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Bio-Researcher AI | Yashwant Nama",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# ==============================
# CONFIG
# ==============================

PDF_PATH = "lehninger.pdf"

# 🔴 USE EXACTLY THIS FORMAT
DROPBOX_URL = "https://www.dropbox.com/scl/fi/wzbf5ra623k6ex3pt98gc/lehninger.pdf?rlkey=fzauw5kna9tyyo2g336f8w5a0&dl=1"

# ==============================
# DOWNLOAD FUNCTION
# ==============================

@st.cache_data(show_spinner=False)
def download_pdf():

    try:
        if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 5_000_000:
            return True

        st.info("📥 Downloading Lehninger PDF...")

        response = requests.get(DROPBOX_URL, stream=True)

        if response.status_code != 200:
            st.error("Failed to download PDF.")
            return False

        with open(PDF_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Verify PDF header
        with open(PDF_PATH, "rb") as f:
            header = f.read(4)
            if header != b"%PDF":
                os.remove(PDF_PATH)
                st.error("Downloaded file is not a valid PDF.")
                return False

        st.success("✅ PDF Ready.")
        return True

    except Exception as e:
        st.error(f"Download error: {e}")
        return False

# ==============================
# VECTOR STORE
# ==============================

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return PineconeVectorStore(
        index_name="lehninger-index",
        embedding=embeddings,
        pinecone_api_key=st.secrets["PINECONE_API_KEY"]
    )

# ==============================
# VISUAL EXTRACTION
# ==============================

def extract_visual(page_num):

    if not os.path.exists(PDF_PATH):
        return "file_not_found"

    try:
        doc = fitz.open(PDF_PATH)
        page = doc.load_page(int(page_num) - 1)

        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        return Image.open(io.BytesIO(pix.tobytes("png")))

    except Exception as e:
        return str(e)

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:
    st.title("👨‍🔬 Researcher Info")
    st.markdown("""
    **Yashwant Nama**  
    PhD Applicant | Molecular Biology  
    Project: Multimodal RAG for Metabolic Research
    """)
    st.divider()

    if st.checkbox("Debug PDF Status"):
        st.write("PDF Exists:", os.path.exists(PDF_PATH))
        if os.path.exists(PDF_PATH):
            st.write("Size (MB):", round(os.path.getsize(PDF_PATH)/1024**2, 2))

# ==============================
# MAIN APP
# ==============================

st.title("🧬 Molecular Biology Research Assistant")

pdf_ready = download_pdf()

if pdf_ready:

    docsearch = load_vectorstore()

    query = st.text_input("Enter your research question:")

    if query:

        results = docsearch.similarity_search(query, k=3)

        for i, doc in enumerate(results):

            page = int(float(doc.metadata.get("page", 0)))

            col1, col2 = st.columns([2,1])

            with col1:
                st.markdown(f"### Result {i+1} | Page {page}")
                st.info(doc.page_content)

            with col2:
                if st.button(f"View Page {page}", key=f"btn_{i}"):

                    img = extract_visual(page)

                    if isinstance(img, Image.Image):
                        st.image(img, use_container_width=True)
                    elif img == "file_not_found":
                        st.error("PDF not found.")
                    else:
                        st.error(img)

            st.divider()

else:
    st.error("PDF could not be loaded.")
