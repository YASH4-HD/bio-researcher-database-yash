import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf as fitz  # PyMuPDF
import requests
import streamlit as st
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================================
# 1️⃣ CONFIGURATION
# ==========================================================

DEFAULT_PDF_NAME = "lehninger.pdf"
UPLOADED_PDF_NAME = "uploaded_lehninger.pdf"
DROPBOX_URL = (
    "https://dl.dropboxusercontent.com/scl/fi/wzbf5ra623k6ex3pt98gc/"
    "lehninger.pdf?rlkey=fzauw5kna9tyyo2g336f8w5a0&dl=1"
)


# ==========================================================
# 2️⃣ PDF DISCOVERY + DOWNLOAD + VISUAL EXTRACTION
# ==========================================================

def find_existing_pdf() -> Optional[str]:
    """Return a readable local PDF path if one is already available."""
    priority = [UPLOADED_PDF_NAME, DEFAULT_PDF_NAME]
    for name in priority:
        if os.path.exists(name) and os.path.getsize(name) > 0:
            return name

    for candidate in Path(".").glob("*.pdf"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)

    return None


def download_pdf() -> Optional[str]:
    """Download the source PDF and return the local path when successful."""
    existing = find_existing_pdf()
    if existing:
        return existing

    try:
        response = requests.get(DROPBOX_URL, timeout=45)
        response.raise_for_status()

        with open(DEFAULT_PDF_NAME, "wb") as file_obj:
            file_obj.write(response.content)

        if os.path.exists(DEFAULT_PDF_NAME) and os.path.getsize(DEFAULT_PDF_NAME) > 0:
            return DEFAULT_PDF_NAME
        return None
    except Exception:
        return None


def save_uploaded_pdf() -> Optional[str]:
    """Persist uploaded PDF to disk and return path."""
    uploaded = st.session_state.get("uploaded_pdf")
    if not uploaded:
        return None

    with open(UPLOADED_PDF_NAME, "wb") as file_obj:
        file_obj.write(uploaded.getbuffer())

    if os.path.exists(UPLOADED_PDF_NAME) and os.path.getsize(UPLOADED_PDF_NAME) > 0:
        return UPLOADED_PDF_NAME
    return None


def resolve_pdf_path() -> Optional[str]:
    """Get an accessible PDF path for visual extraction."""
    uploaded_path = save_uploaded_pdf()
    if uploaded_path:
        return uploaded_path
    return download_pdf()


def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def page_visual_density(page) -> int:
    """Simple signal for whether a page has figures/diagrams."""
    drawings = page.get_drawings()
    images = page.get_image_info()
    return len(drawings) + len(images)


def search_image_pages(pdf_path: str, query: str, k: int = 6) -> List[Dict[str, object]]:
    """Search pages that are most likely to contain visuals relevant to query text."""
    doc = fitz.open(pdf_path)
    query_norm = normalize_text(query)
    query_tokens = [token for token in query_norm.split() if len(token) > 3]

    if not query_tokens:
        return []

    candidates: List[Dict[str, object]] = []

    for idx in range(len(doc)):
        page = doc.load_page(idx)
        density = page_visual_density(page)
        if density == 0:
            continue

        page_text = normalize_text(page.get_text("text"))
        overlap = sum(1 for token in set(query_tokens) if token in page_text)

        if overlap == 0:
            continue

        score = (overlap * 10) + min(density, 10)
        snippet = page.get_text("text")[:350].strip().replace("\n", " ")
        candidates.append(
            {
                "page": idx + 1,
                "score": score,
                "visual_density": density,
                "snippet": snippet,
            }
        )

    candidates.sort(key=lambda item: (item["score"], item["visual_density"]), reverse=True)
    return candidates[:k]


def extract_smart_visuals(page_num, pdf_path: str, mode="Smart Crop"):
    try:
        if not pdf_path or not os.path.exists(pdf_path):
            return "file_not_found"

        doc = fitz.open(pdf_path)
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

    except Exception as error:
        return str(error)


# ==========================================================
# 3️⃣ VECTOR STORE LOADER
# ==========================================================

@st.cache_resource
def load_vectorstore():
    from langchain_pinecone import PineconeVectorStore

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
    lowered = (query or "").lower()

    signals = {
        "enzymes": any(keyword in lowered for keyword in ["enzyme", "transferase", "kinase", "hydrolase"]),
        "pathways": any(keyword in lowered for keyword in ["pathway", "glycolysis", "cycle", "metabolism"]),
        "disease": any(keyword in lowered for keyword in ["disease", "mutation", "cancer", "defect"]),
        "lab": any(keyword in lowered for keyword in ["experiment", "protocol", "assay", "lab"]),
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

    st.file_uploader(
        "Optional: Upload Lehninger PDF for diagram extraction",
        type=["pdf"],
        key="uploaded_pdf",
        help="Use this if auto-download fails or if you want to use a local copy.",
    )

    st.divider()
    if st.checkbox("Show Debug Info"):
        detected_pdf = find_existing_pdf()
        st.write("Detected PDF:", detected_pdf or "None")
        if detected_pdf:
            st.write("File Size (MB):", round(os.path.getsize(detected_pdf) / 1024**2, 2))


# ==========================================================
# 7️⃣ MAIN UI
# ==========================================================

st.title("🧬 Molecular Biology Research Assistant")
st.caption("AI-powered knowledge retrieval from Lehninger Principles of Biochemistry")

pdf_path = resolve_pdf_path()

if not pdf_path:
    st.warning(
        "⚠️ Diagram extraction PDF is missing. You can still query text results, "
        "but to extract visuals please upload a PDF in the sidebar."
    )

text_tab, image_tab, ideas_tab = st.tabs([
    "🔎 Text Research",
    "🖼️ Image Explorer",
    "🚀 Unique Feature Additions",
])

with text_tab:
    st.subheader("Text Research")
    query = st.text_input("Enter your research question:", placeholder="e.g. Describe transferases", key="text_query")

    if query:
        with st.spinner("🔬 Searching metabolic database..."):
            try:
                docsearch = load_vectorstore()
                results = docsearch.similarity_search(query, k=3)
            except Exception as error:
                st.error("Unable to connect to Pinecone vector store. Please verify dependencies and credentials.")
                st.code(str(error))
                st.info("If this is a package-rename issue, remove `pinecone-client` and install `pinecone`.")
                results = []

            if not results:
                st.warning("No matches found in vector index.")

            for index, doc in enumerate(results):
                raw_page = doc.metadata.get("page", 0)
                clean_page = int(float(raw_page))
                st.markdown(f"### Result {index + 1} | Metadata Page {clean_page}")
                st.info(doc.page_content)
                st.caption("Visual extraction is handled in the **Image Explorer** tab to avoid text/visual mismatch.")
                st.divider()

with image_tab:
    st.subheader("Image Explorer (visual-first search)")
    st.caption("Search directly for pages with diagrams/figures, independent of text-result metadata.")

    image_query = st.text_input(
        "Search image content by topic:",
        placeholder="e.g. CRISPR Cas9 mechanism",
        key="image_query",
    )

    max_hits = st.slider("Number of image pages to return", min_value=1, max_value=12, value=6)

    if st.button("Search Image Pages", key="image_search_btn"):
        if not pdf_path:
            st.error("PDF unavailable. Upload Lehninger PDF in sidebar to search image pages.")
        elif not image_query.strip():
            st.warning("Please enter a search topic for image pages.")
        else:
            with st.spinner("Scanning PDF pages with visuals..."):
                matches = search_image_pages(pdf_path, image_query, k=max_hits)

            if not matches:
                st.warning("No visual-heavy pages matched your query. Try broader keywords.")

            for idx, match in enumerate(matches):
                page_num = int(match["page"])
                score = match["score"]
                density = match["visual_density"]

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### Visual Match {idx + 1} | Page {page_num}")
                    st.markdown(f"- Match score: **{score}**")
                    st.markdown(f"- Visual density: **{density}**")
                    st.info(match["snippet"] or "No preview text available for this page.")

                with col2:
                    img = extract_smart_visuals(page_num, pdf_path, extraction_mode)
                    if isinstance(img, Image.Image):
                        st.image(
                            img,
                            use_container_width=True,
                            caption=f"Image Explorer source page: {page_num}",
                        )
                    elif img == "file_not_found":
                        st.error("PDF file not found. Re-upload PDF from sidebar.")
                    else:
                        st.error(f"Extraction failed: {img}")

                st.divider()

with ideas_tab:
    st.subheader("Unique Feature Additions Explorer")
    ideas_query = st.text_input(
        "Enter a topic to brainstorm new capabilities:",
        placeholder="e.g. pathway mutation analysis",
        key="ideas_query",
    )

    if st.button("Generate Unique Feature Additions", key="ideas_btn"):
        for concept in generate_unique_feature_additions(ideas_query):
            st.markdown(f"### {concept['name']}")
            st.markdown(f"- **Why it is unique:** {concept['uniqueness']}")
            st.markdown(f"- **Research value:** {concept['value']}")
            st.markdown(f"- **Prototype direction:** {concept['prototype']}")
            st.divider()
