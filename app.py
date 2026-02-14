import streamlit as st
import os
import fitz
from PIL import Image
import io
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ==============================
# 1. CONFIGURATION
# ==============================

PDF_PATH = "lehninger.pdf"

# 🔴 Replace with converted dropbox direct link
DROPBOX_URL = "https://dl.dropboxusercontent.com/scl/fi/wzbf5ra623k6ex3pt98gc/lehninger.pdf?rlkey=fzauw5kna9tyyo2g336f8w5a0
"

# ==============================
# 2. SAFE DROPBOX DOWNLOADER
# ==============================

@st.cache_data(show_spinner=False)
def download_pdf():
    """Force-download PDF and verify integrity."""
    
    try:
        if os.path.exists(PDF_PATH):
            # Check file size to ensure it's real
            if os.path.getsize(PDF_PATH) > 5_000_000:  # >5MB
                return True
            else:
                os.remove(PDF_PATH)  # Corrupted file, delete it

        st.info("📥 Downloading Lehninger PDF...")

        with urllib.request.urlopen(DROPBOX_URL) as response:
            data = response.read()

            # Check if Dropbox returned HTML instead of PDF
            if b"<html" in data[:500]:
                st.error("Dropbox returned HTML instead of PDF. Check link.")
                return False

            with open(PDF_PATH, "wb") as f:
                f.write(data)

        # Final verification
        if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 5_000_000:
            st.success("✅ PDF successfully downloaded and verified.")
            return True
        else:
            st.error("Downloaded file is invalid.")
            return False

    except Exception as e:
        st.error(f"Download error: {e}")
        return False

            # Write in chunks (important for large PDFs)
            with open(PDF_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Validate file
            try:
                fitz.open(PDF_PATH)
                st.success("✅ PDF ready.")
                return True
            except:
                os.remove(PDF_PATH)
                st.error("Downloaded file is corrupted.")
                return False

    except Exception as e:
        st.error(f"Download error: {e}")
        return False

            # Validate PDF header
            if not response.content.startswith(b"%PDF"):
                st.error("Downloaded file is not a valid PDF. Check Dropbox link.")
                return False

            with open(PDF_PATH, "wb") as f:
                f.write(response.content)

            st.success("✅ Database ready!")

            return True

    except Exception as e:
        st.error(f"Download error: {e}")
        return False


# ==============================
# 3. VECTOR STORE LOADER
# ==============================

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name="lehninger-index",
        embedding=embeddings,
        pinecone_api_key=st.secrets["PINECONE_API_KEY"]
    )
    return vectorstore


# ==============================
# 4. VISUAL EXTRACTION
# ==============================

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


# ==============================
# 5. STREAMLIT UI
# ==============================

st.set_page_config(
    page_title="Bio-Researcher AI | Yashwant Nama",
    layout="wide",
    page_icon="🧬"
)

# Sidebar
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

# Main Title
st.title("🧬 Molecular Biology Research Assistant")
st.caption("Quantitative AI-Powered Research Interface")

# ==============================
# 6. APP LOGIC
# ==============================

pdf_ready = download_pdf()

if pdf_ready:

    docsearch = load_vectorstore()
    query = st.text_input(
        "Enter your research question:",
        placeholder="e.g. Describe transferases"
    )

    if query:
        with st.spinner("Analyzing molecular pathways..."):

            results = docsearch.similarity_search(query, k=3)

            if not results:
                st.warning("No matches found.")

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
                                st.image(img, use_container_width=True)
                            elif img == "file_not_found":
                                st.error("PDF not found.")
                            else:
                                st.error(f"Extraction failed: {img}")

                st.divider()

else:
    st.error("PDF could not be loaded. Check Dropbox link.")
