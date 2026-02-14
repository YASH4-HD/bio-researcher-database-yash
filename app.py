import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    priority = [UPLOADED_PDF_NAME, DEFAULT_PDF_NAME]
    for name in priority:
        if os.path.exists(name) and os.path.getsize(name) > 0:
            return name

    for candidate in Path(".").glob("*.pdf"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)

    return None


def download_pdf() -> Optional[str]:
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
    uploaded = st.session_state.get("uploaded_pdf")
    if not uploaded:
        return None

    with open(UPLOADED_PDF_NAME, "wb") as file_obj:
        file_obj.write(uploaded.getbuffer())

    if os.path.exists(UPLOADED_PDF_NAME) and os.path.getsize(UPLOADED_PDF_NAME) > 0:
        return UPLOADED_PDF_NAME
    return None


def resolve_pdf_path() -> Optional[str]:
    uploaded_path = save_uploaded_pdf()
    if uploaded_path:
        return uploaded_path
    return download_pdf()


def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def page_visual_density(page) -> int:
    drawings = page.get_drawings()
    images = page.get_image_info()
    return len(drawings) + len(images)


def search_image_pages(pdf_path: str, query: str, k: int = 6) -> List[Dict[str, object]]:
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
# 4️⃣ LEARNING ENGINES HELPERS
# ==========================================================


def genetic_circuit_outcome(lox1: str, lox2: str, has_promoter: bool, has_stop: bool) -> Tuple[str, str]:
    if lox1 == lox2:
        recomb = "Deletion between Lox sites"
        expression = "ON" if has_promoter and not has_stop else "OFF"
    else:
        recomb = "Inversion between Lox sites"
        expression = "Conditional/Orientation-dependent"
    explanation = (
        f"Recombination: **{recomb}**. Predicted expression: **{expression}**. "
        "This is a deterministic rule-based prediction from site orientation + cassette context."
    )
    return recomb, explanation


def parse_edges(edge_text: str) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    activations: Set[Tuple[str, str]] = set()
    inhibitions: Set[Tuple[str, str]] = set()
    for line in edge_text.splitlines():
        s = line.strip().replace(" ", "")
        if not s:
            continue
        if "->" in s:
            a, b = s.split("->", 1)
            activations.add((a, b))
        elif "-|" in s:
            a, b = s.split("-|", 1)
            inhibitions.add((a, b))
    return activations, inhibitions


def run_pathway_simulation(nodes: List[str], active_nodes: Set[str], activations, inhibitions, steps: int = 5):
    states = []
    current = {n: (n in active_nodes) for n in nodes}
    states.append(current.copy())

    for _ in range(steps):
        nxt = current.copy()
        for node in nodes:
            act_signal = any(current.get(src, False) for src, dst in activations if dst == node)
            inh_signal = any(current.get(src, False) for src, dst in inhibitions if dst == node)
            if inh_signal:
                nxt[node] = False
            elif act_signal:
                nxt[node] = True
        current = nxt
        states.append(current.copy())
    return states


def pcr_expected_size(start: int, end: int) -> int:
    return abs(end - start) + 1


def michaelis_menten_curve(vmax: float, km: float):
    x = [i / 10 for i in range(0, 101)]
    y = [vmax * s / (km + s) if (km + s) != 0 else 0 for s in x]
    return x, y


def hardy_weinberg(p: float):
    q = 1 - p
    return p * p, 2 * p * q, q * q


def logic_breakdown(question: str) -> List[str]:
    q = question.lower()
    steps = ["Identify domain and entities in the question."]

    if any(k in q for k in ["pedigree", "inherit", "trait"]):
        steps.append("Infer inheritance model (autosomal/recessive/dominant/X-linked).")
    if any(k in q for k in ["pathway", "activate", "inhibit"]):
        steps.append("Construct causal network and propagate activation/inhibition rules.")
    if any(k in q for k in ["km", "vmax", "kinetics", "enzyme"]):
        steps.append("Map symbols to equation, substitute values, and interpret curve-shift behavior.")
    if any(k in q for k in ["pcr", "band", "western", "facs"]):
        steps.append("Convert assay conditions into measurable outputs (band size/intensity/peak shift).")

    steps.append("Generate final answer with assumptions + confidence note.")
    return steps


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

text_tab, image_tab, engines_tab = st.tabs(["🔎 Text Research", "🖼️ Image Explorer", "🧠 Learning Engines"])

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

    image_query = st.text_input("Search image content by topic:", placeholder="e.g. CRISPR Cas9 mechanism", key="image_query")
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

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### Visual Match {idx + 1} | Page {page_num}")
                    st.markdown(f"- Match score: **{match['score']}**")
                    st.markdown(f"- Visual density: **{match['visual_density']}**")
                    st.info(match["snippet"] or "No preview text available for this page.")

                with col2:
                    img = extract_smart_visuals(page_num, pdf_path, extraction_mode)
                    if isinstance(img, Image.Image):
                        st.image(img, use_container_width=True, caption=f"Image Explorer source page: {page_num}")
                    elif img == "file_not_found":
                        st.error("PDF file not found. Re-upload PDF from sidebar.")
                    else:
                        st.error(f"Extraction failed: {img}")

                st.divider()

with engines_tab:
    st.subheader("Learning Engines for Practical Skills")
    st.caption("Five dedicated tool-tabs for genetics, signaling, experiments, equations, and reasoning.")

    genetic_tab, pathway_tab, experiment_tab, equation_tab, logic_tab = st.tabs(
        [
            "1️⃣ Genetic Circuit",
            "2️⃣ Pathway Simulator",
            "3️⃣ Experimental Predictor",
            "4️⃣ Equation Solver",
            "5️⃣ Logic-Inference",
        ]
    )

    with genetic_tab:
        st.markdown("### Genetic Circuit Engine")
        lox1 = st.selectbox("Lox Site 1 orientation", ["same", "reverse"], key="lox1")
        lox2 = st.selectbox("Lox Site 2 orientation", ["same", "reverse"], key="lox2")
        has_promoter = st.checkbox("Promoter present", value=True)
        has_stop = st.checkbox("STOP cassette present", value=False)
        if st.button("Run Genetic Circuit", key="run_genetic"):
            recomb, msg = genetic_circuit_outcome(lox1, lox2, has_promoter, has_stop)
            st.success(msg)
            st.write(f"Recommended interpretation for exam answer: **{recomb}** + effect on gene expression.")

    with pathway_tab:
        st.markdown("### Pathway Simulator")
        nodes_text = st.text_input("Nodes (comma separated)", "A,B,C,D", key="nodes_text")
        edge_text = st.text_area(
            "Edges (one per line: A->B for activation, A-|B for inhibition)",
            "A->B\nB-|C\nA->D",
            key="edge_text",
            height=130,
        )
        active_text = st.text_input("Initially active nodes (comma separated)", "A", key="active_nodes")
        steps = st.slider("Simulation steps", min_value=1, max_value=10, value=4)

        if st.button("Simulate Pathway", key="run_pathway"):
            nodes = [n.strip() for n in nodes_text.split(",") if n.strip()]
            activations, inhibitions = parse_edges(edge_text)
            active_nodes = {n.strip() for n in active_text.split(",") if n.strip()}
            states = run_pathway_simulation(nodes, active_nodes, activations, inhibitions, steps)

            st.write("**State evolution:**")
            for i, state in enumerate(states):
                on_nodes = [n for n, v in state.items() if v]
                st.write(f"Step {i}: {' , '.join(on_nodes) if on_nodes else 'No active nodes'}")

    with experiment_tab:
        st.markdown("### Experimental Result Predictor")
        assay = st.selectbox("Assay type", ["PCR", "Western Blot", "FACS"]) 

        if assay == "PCR":
            start = st.number_input("Forward primer start", min_value=1, value=120)
            end = st.number_input("Reverse primer end", min_value=1, value=780)
            if st.button("Predict PCR Product"):
                size = pcr_expected_size(int(start), int(end))
                st.success(f"Expected amplicon size: **{size} bp**")

        elif assay == "Western Blot":
            proteins = st.text_input("Protein bands in kDa (comma separated)", "42,55,110")
            if st.button("Generate Synthetic Blot Profile"):
                vals = [float(x.strip()) for x in proteins.split(",") if x.strip()]
                chart_data = {"kDa": vals, "Intensity": [1.0 - (i * 0.15) for i in range(len(vals))]}
                st.write("Predicted band intensities")
                st.bar_chart(chart_data, x="kDa", y="Intensity")

        else:
            marker1 = st.number_input("Control median fluorescence", min_value=0.0, value=120.0)
            marker2 = st.number_input("Treated median fluorescence", min_value=0.0, value=220.0)
            if st.button("Predict FACS Shift"):
                delta = marker2 - marker1
                st.success(f"Predicted fluorescence shift: **{delta:.2f} units**")
                st.line_chart({"control": [marker1] * 20, "treated": [marker2] * 20})

    with equation_tab:
        st.markdown("### Equation Solver + Grapher")
        eq_type = st.selectbox("Equation model", ["Michaelis-Menten", "Hardy-Weinberg", "Gibbs Free Energy"])

        if eq_type == "Michaelis-Menten":
            vmax = st.number_input("Vmax", min_value=0.01, value=2.0)
            km = st.number_input("Km", min_value=0.01, value=1.0)
            if st.button("Plot Kinetics"):
                x, y = michaelis_menten_curve(vmax, km)
                st.line_chart({"[S]": x, "v": y}, x="[S]", y="v")
                st.info("Interpretation: lower Km shifts curve left (higher affinity).")

        elif eq_type == "Hardy-Weinberg":
            p = st.slider("Allele frequency p", min_value=0.0, max_value=1.0, value=0.6)
            if st.button("Compute Genotype Frequencies"):
                p2, two_pq, q2 = hardy_weinberg(p)
                st.write(f"p² = {p2:.3f}, 2pq = {two_pq:.3f}, q² = {q2:.3f}")
                st.bar_chart({"frequency": [p2, two_pq, q2]}, x=None, y="frequency")

        else:
            delta_h = st.number_input("ΔH (kJ/mol)", value=-40.0)
            temp_k = st.number_input("Temperature (K)", min_value=1.0, value=298.0)
            delta_s = st.number_input("ΔS (kJ/mol·K)", value=-0.1)
            if st.button("Solve ΔG"):
                delta_g = delta_h - (temp_k * delta_s)
                spontaneity = "Spontaneous" if delta_g < 0 else "Non-spontaneous"
                st.success(f"ΔG = {delta_g:.3f} kJ/mol → **{spontaneity}**")

    with logic_tab:
        st.markdown("### Logic-Inference Chatbot (Step-by-Step)")
        q = st.text_area("Paste a complex Part-C style question", height=140)
        if st.button("Generate Logic Steps"):
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                steps = logic_breakdown(q)
                st.write("**Reasoning plan:**")
                for i, step in enumerate(steps, start=1):
                    st.markdown(f"{i}. {step}")

                if st.checkbox("Attach RAG evidence snippets"):
                    try:
                        docsearch = load_vectorstore()
                        evidence = docsearch.similarity_search(q, k=2)
                        st.write("**Supporting snippets:**")
                        for item in evidence:
                            st.info(item.page_content[:400])
                    except Exception as error:
                        st.error("Could not fetch RAG evidence.")
                        st.code(str(error))
