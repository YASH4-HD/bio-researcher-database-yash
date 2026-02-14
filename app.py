import io
import os
from typing import Dict, List

import fitz  # PyMuPDF
import requests
import streamlit as st
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ==========================================================
# 1️⃣ CONFIGURATION
# ==========================================================

PDF_PATH = "lehninger.pdf"
DROPBOX_URL = (
    "https://dl.dropboxusercontent.com/scl/fi/wzbf5ra623k6ex3pt98gc/"
    "lehninger.pdf?rlkey=fzauw5kna9tyyo2g336f8w5a0&dl=1"
)
# ==========================================================
# 2️⃣ PDF DOWNLOADER + VISUAL EXTRACTION
# ==========================================================

def download_pdf() -> bool:
    """Download the source PDF once and keep it locally for visual extraction."""
    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0:
        return True

    try:
        response = requests.get(DROPBOX_URL, timeout=30)
        response.raise_for_status()

        with open(PDF_PATH, "wb") as f:
            f.write(response.content)

        return os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0
    except Exception:
        return False


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
                visual_rect = bboxes[0]
                for bbox in bboxes[1:]:
                    visual_rect = visual_rect | bbox
                page.set_cropbox(visual_rect + (-15, -15, 15, 15))

        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
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
        pinecone_api_key=st.secrets["PINECONE_API_KEY"],
    )
    return vectorstore


# ==========================================================
# 4️⃣ UNIQUE FEATURE IDEATION HELPERS
# ==========================================================

def generate_unique_feature_additions(query: str) -> List[Dict[str, str]]:
    """Generate practical, unique product-feature ideas contextualized to the query."""
    lowered = (query or "").lower()

    signals = {
        "enzymes": any(k in lowered for k in ["enzyme", "transferase", "kinase", "hydrolase"]),
        "pathways": any(k in lowered for k in ["pathway", "glycolysis", "cycle", "metabolism"]),
        "disease": any(k in lowered for k in ["disease", "mutation", "cancer", "defect"]),
        "lab": any(k in lowered for k in ["experiment", "protocol", "assay", "lab"]),
    }

    concepts = [
        {
            "name": "Mechanism Contrast Mode",
            "uniqueness": "Compares competing biochemical mechanisms side-by-side from retrieved context.",
            "value": "Helps researchers quickly spot agreement gaps and uncertain reaction steps.",
            "prototype": "Add dual evidence panels with confidence bars based on chunk overlap.",
        },
        {
            "name": "Pathway Perturbation Simulator",
            "uniqueness": "Lets users apply in-silico perturbations (enzyme inhibition/overexpression) to conceptual pathway flow.",
            "value": "Useful for forming stronger hypotheses before wet-lab work.",
            "prototype": "Create editable pathway nodes and estimate downstream qualitative impact labels.",
        },
        {
            "name": "Disease Mutation Lens",
            "uniqueness": "Bridges core biochemistry passages to likely mutation-sensitive pathway points.",
            "value": "Improves translational relevance for clinical or biotech-focused questions.",
            "prototype": "Tag retrieval results with mutation-risk annotations and likely phenotype categories.",
        },
        {
            "name": "Experiment Starter Builder",
            "uniqueness": "Auto-drafts a minimal experiment plan from retrieved evidence and user goal.",
            "value": "Speeds up proposal drafting and improves reproducibility.",
            "prototype": "Generate sections for objective, controls, readout, and failure modes.",
        },
    ]

    if signals["pathways"]:
        concepts[1]["value"] += " Especially strong for metabolism/pathway questions."
    if signals["disease"]:
        concepts[2]["value"] += " Prioritizes medically relevant hypotheses."
    if signals["lab"]:
        concepts[3]["prototype"] += " Include time/cost estimate sliders for lab planning."
    if signals["enzymes"]:
        concepts[0]["prototype"] += " Highlight catalytic-site specific evidence snippets."

    return concepts


# ==========================================================
# 5️⃣ STREAMLIT PAGE SETUP
# ==========================================================

st.set_page_config(page_title="Bio-Researcher AI | Yashwant Nama", layout="wide", page_icon="🧬")


# ==========================================================
# 6️⃣ SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("👨‍🔬 Researcher Info")
    st.markdown(
        """
    **Yashwant Nama**  
    PhD Applicant | Molecular Biology  

    Project: Multimodal RAG for Metabolic Research
    """
    )
    st.divider()

    extraction_mode = st.radio("Visual Extraction Mode:", ["Smart Crop", "Full Page View"])

    st.divider()
    if st.checkbox("Show Debug Info"):
        st.write("PDF Exists:", os.path.exists(PDF_PATH))
        if os.path.exists(PDF_PATH):
            st.write("File Size (MB):", round(os.path.getsize(PDF_PATH) / 1024**2, 2))


# ==========================================================
# 7️⃣ MAIN UI
# ==========================================================

st.title("🧬 Molecular Biology Research Assistant")
st.caption("AI-powered knowledge retrieval from Lehninger Principles of Biochemistry")

pdf_ready = download_pdf()

if pdf_ready:
    query = st.text_input("Enter your research question:", placeholder="e.g. Describe transferases")

    with st.expander("🚀 Unique Feature Additions Explorer", expanded=bool(query)):
        st.write(
            "Explore high-impact product ideas tailored to your active research question."
        )
        if st.button("Generate Unique Feature Additions"):
            for concept in generate_unique_feature_additions(query):
                st.markdown(f"### {concept['name']}")
                st.markdown(f"- **Why it is unique:** {concept['uniqueness']}")
                st.markdown(f"- **Research value:** {concept['value']}")
                st.markdown(f"- **Prototype direction:** {concept['prototype']}")
                st.divider()

    if query:
        with st.spinner("🔬 Searching metabolic database..."):
            docsearch = load_vectorstore()
            results = docsearch.similarity_search(query, k=3)

            if not results:
                st.warning("No matches found in vector index.")

            for i, doc in enumerate(results):
                raw_page = doc.metadata.get("page", 0)
                clean_page = int(float(raw_page))

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown(f"### Result {i + 1} | Page {clean_page}")
                    st.info(doc.page_content)

                with col2:
                    if st.button(f"🔍 Extract Visuals (P. {clean_page})", key=f"btn_{i}"):
                        with st.spinner("Extracting diagrams..."):
                            img = extract_smart_visuals(clean_page, extraction_mode)

                            if isinstance(img, Image.Image):
                                st.image(
                                    img,
                                    use_container_width=True,
                                    caption=f"Source: Page {clean_page}",
                                )
                            elif img == "file_not_found":
                                st.error("PDF file not found on server.")
                            else:
                                st.error(f"Extraction failed: {img}")

                st.divider()

else:
    st.error("❌ PDF could not be loaded. Check Dropbox link.")
