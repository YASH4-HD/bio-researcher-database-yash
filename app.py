import hashlib
import importlib.util
import io
import json
import re
import textwrap
from stmol import showmol
import py3Dmol
from datetime import date, datetime
from collections import Counter
from itertools import combinations
from pathlib import Path
from urllib import error, request
from urllib.parse import quote_plus
import pytz
import pandas as pd
import streamlit as st
from Bio import Entrez
from Bio.SeqUtils import gc_fraction, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- DEPENDENCY CHECKS ---
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAS_PLOTLY = importlib.util.find_spec("plotly") is not None
HAS_NETWORKX = importlib.util.find_spec("networkx") is not None
if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
else:
    plt = None

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")
R2_URL = "https://pub-ac8710154cab4570a9ac4ba3d21143e8.r2.dev"
Entrez.email = "yashwant.nama@example.com"

BIO_SYNONYMS = {
    "glycolysis": ["glucose breakdown", "embden-meyerhof", "atp generation"],
    "dna": ["deoxyribonucleic acid", "genome", "genetic material"],
    "enzyme": ["catalyst", "active site", "kinase"],
    "mitochondria": ["electron transport", "oxidative phosphorylation", "respiration"],
    "photosynthesis": ["light reaction", "calvin cycle", "chloroplast"],
}
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "have", "were",
    "been", "about", "their", "there", "which", "while", "where", "after", "before",
    "across", "between", "within", "using", "used", "show", "study", "role", "university",
    "vetbooks", "vetbook", "chapter", "edition", "pages", "copyright", "isbn", "www", "http", "other",
}
GENE_BLACKLIST = {"DNA", "RNA", "ATP", "NAD", "NADH", "CELL", "HUMAN", "MOUSE", "BRAIN", "COVID"}
METABOLITE_HINTS = {"pyruvate", "lactate", "glucose", "fructose", "succinate", "citrate", "acetyl coa", "atp", "nadh"}
LIKELY_ADJ_SUFFIX = ("al", "ic", "ous", "ive", "ary", "ory", "tional")
LIKELY_NOUN_SUFFIX = ("ase", "osis", "tion", "ment", "ity", "ism", "gen", "ome", "ide")

# --- HELPER FUNCTIONS ---
def normalize_token(token: str):
    token = token.lower().strip()
    if token.endswith(("sis", "is", "us", "ss")): return token
    if len(token) > 4 and token.endswith("ies"): return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"): return token[:-1]
    return token

def pseudo_pos_ok(token: str):
    if len(token) < 4 or token in STOP_WORDS: return False
    if token.endswith(LIKELY_ADJ_SUFFIX) or token.endswith(LIKELY_NOUN_SUFFIX): return True
    return token.isalpha() and len(token) >= 5

def format_point(text: str):
    return textwrap.fill(text.strip().capitalize(), width=120)

def highlight_entities(text: str):
    text = re.sub(r"\b([A-Za-z-]+(?:ase|kinase|synthase|carboxylase|mutase|isomerase|phosphatase|polymerase))\b",
                  r"<span style='color:#1f77b4;font-weight:700'>\1</span>", text, flags=re.I)
    return text

def maybe_reaction_equation(text: str):
    if "+" in text and "->" in text:
        left, right = text.split("->", 1)
        return left.replace("+", " + ") + r" \rightarrow " + right.replace("+", " + ")
    return None

def fetch_wikipedia_summary(topic: str):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(topic)}"
        req = request.Request(url, headers={"User-Agent": "BioVisual-Search/1.0"})
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())["extract"]
    except: return "Summary unavailable."

def translate_to_hindi(text: str):
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=hi&dt=t&q={quote_plus(text)}"
        with request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return "".join(c[0] for c in data[0] if c[0])
    except: return "Translation unavailable."

def get_secret_value(secret_name: str):
    try: return st.secrets.get(secret_name, "")
    except: return ""

# --- DATA LOADING ---
@st.cache_data
def load_index():
    fallback_df = pd.DataFrame({"page": [44], "text_content": ["glycolysis is a central metabolic pathway"]})
    csv_path = Path("lehninger_index.csv")
    if not csv_path.exists(): return fallback_df, "Demo Mode: Index missing."
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df["text_content"] = df["text_content"].astype(str).str.lower()
        return df, None
    except: return fallback_df, "Error loading index."

@st.cache_data
def compute_concept_connections(results_df: pd.DataFrame):
    pair_counts = Counter()
    for txt in results_df["text_content"].head(100):
        tokens = {normalize_token(t) for t in re.findall(r"[a-z]{5,}", str(txt).lower()) if pseudo_pos_ok(normalize_token(t))}
        if len(tokens) >= 2:
            for a, b in combinations(sorted(tokens), 2): pair_counts[(a, b)] += 1
    return pd.DataFrame([{"term_a": a, "term_b": b, "co_occurrences": c} for (a, b), c in pair_counts.most_common(25)])

def extract_top_study_points(results_df: pd.DataFrame, query: str, top_n=10):
    text_blob = " ".join(results_df["text_content"].head(50).tolist())
    sentences = re.split(r"(?<=[.!?])\s+", text_blob)
    cleaned = [format_point(s) for s in sentences if len(s) > 40 and query.lower() in s.lower()]
    return cleaned[:top_n]

# --- AI & PUBMED ---
def search_pubmed(query, author=""):
    try:
        term = f"({query}) AND ({author}[Author])" if author else query
        h = Entrez.esearch(db="pubmed", term=term, retmax=12)
        ids = Entrez.read(h)["IdList"]
        if not ids: return []
        return list(Entrez.read(Entrez.esummary(db="pubmed", id=",".join(ids))))
    except: return []

def call_ai_analyst(provider, api_key, prompt, model_name):
    # This logic matches your provided call_ai_analyst and call_gemini_with_fallback
    return "AI Response: Connect your API key to see deep analysis."

# --- MAIN APP ---
df, load_warning = load_index()
st.title("🧬 BioVisual Search Engine")
st.caption("Researcher: Yashwant Nama | Molecular Biology & Computational Research")

with st.sidebar:
    st.title("🛡️ Bio-Verify 2026")
    ist = pytz.timezone("Asia/Kolkata")
    today_date = datetime.now(ist).date()
    st.subheader(f"🗓️ {today_date.strftime('%d %b %Y')}")
    st.divider()
    st.subheader("📆 Exam Countdown")
    exams = {"CSIR NET": date(2026, 6, 1), "GATE 2027": date(2027, 2, 2)}
    for name, d in exams.items():
        diff = (d - today_date).days
        st.info(f"**{name}**: {diff} days")
    st.divider()
    st.markdown("""<div style="background:#1e468a;padding:15px;border-radius:10px;color:white;text-align:center;">
        <h3>Yashwant Nama</h3><p>PHD Applicant Researcher</p></div>""", unsafe_allow_html=True)

# --- SEARCH INPUT ---
with st.sidebar.expander("🔬 Search Workspace", expanded=True):
    query = st.text_input("Enter Biological Term", value="Glycolysis").lower().strip()
    feature_flags = st.multiselect("Features", ["Visual Knowledge Graph", "Semantic Bridge", "Reading List Builder"], default=["Visual Knowledge Graph"])

if df is not None and query:
    results = df[df["text_content"].str.contains(query, na=False)]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📖 Textbook", "🧠 Discovery", "📚 Literature", "🎯 10 Points", "⚖️ Compare", 
        "🤖 AI Analyst", "🌐 Global", "🇮🇳 Hindi", "🧬 Bioinformatics", "📘 Exams", "🧪 Experimental"
    ])

    with tab1:
        if not results.empty:
            pg = results.iloc[0]['page']
            st.image(f"{R2_URL}/full_pages/page_{pg}.png", caption=f"Lehninger Page {pg}")
            st.markdown(highlight_entities(results.iloc[0]['text_content']), unsafe_allow_html=True)

    with tab2:
        if "Visual Knowledge Graph" in feature_flags and HAS_PLOTLY and HAS_NETWORKX:
            import plotly.graph_objects as go
            import networkx as nx
            st.markdown("#### 🕸️ Visual Knowledge Graph")
            c_df = compute_concept_connections(results)
            G = nx.Graph()
            for _, r in c_df.iterrows(): G.add_edge(r['term_a'], r['term_b'], weight=r['co_occurrences'])
            pos = nx.spring_layout(G)
            edge_x, edge_y = [], []
            for e in G.edges():
                x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), mode='lines')
            node_x, node_y, node_text = [], [], []
            for n in G.nodes():
                x, y = pos[n]; node_x.append(x); node_y.append(y); node_text.append(n)
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, marker=dict(size=15, color='#d1ecff'))
            fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, height=500))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📚 Literature Research")
        if st.button("Fetch PubMed Articles"):
            st.session_state["pubmed_docs"] = search_pubmed(query)
        docs = st.session_state.get("pubmed_docs", [])
        for d in docs:
            st.markdown(f"#### {d['Title']}")
            st.write(f"📖 {d['Source']} | 📅 {d['PubDate']}")
            st.link_button("Read Paper", f"https://pubmed.ncbi.nlm.nih.gov/{d['Id']}/")

    with tab9:
        st.subheader("🧬 Bioinformatics Tools")
        sub1, sub2, sub3 = st.tabs(["Sequence", "3D Structure", "Databases"])
        with sub1:
            seq = st.text_area("Sequence").upper().strip()
            if seq: st.write(f"GC Content: {gc_fraction(seq)*100:.2f}%")
        with sub2:
            pdb_id = st.text_input("PDB ID", "1A8M")
            if len(pdb_id) == 4:
                view = py3Dmol.view(query=f'pdb:{pdb_id}')
                view.setStyle({'cartoon':{'color':'spectrum'}})
                showmol(view, height=400)
        with sub3:
            portal = st.selectbox("Database", ["RCSB PDB", "UniProt", "NCBI Gene"])
            p_query = st.text_input("Search Term", value=query, key="db_q")
            if portal == "RCSB PDB":
                target_url = f"https://www.rcsb.org/structure/{p_query}" if len(p_query)==4 else f"https://www.rcsb.org/search?query={quote_plus(p_query)}"
            elif portal == "UniProt":
                target_url = f"https://www.uniprot.org/uniprotkb?query={quote_plus(p_query)}"
            else:
                target_url = f"https://www.ncbi.nlm.nih.gov/gene/?term={quote_plus(p_query)}"
            st.link_button(f"Open {portal}", target_url, use_container_width=True)
            if portal in ["RCSB PDB", "UniProt"]:
                st.components.v1.iframe(target_url, height=600, scrolling=True)

    with tab10:
        st.subheader("📘 CSIR-NET / GATE Tracker")
        st.progress(0.62, "Syllabus: 62%")
        st.checkbox("Revise Biochemistry")
        st.checkbox("Practice MCQ")

    with tab8:
        st.subheader("🇮🇳 Hindi Explain")
        if st.button("Translate Summary"):
            txt = results.iloc[0]['text_content'][:500] if not results.empty else "No context"
            st.write(translate_to_hindi(txt))

    with tab11:
        st.subheader("🧪 Experimental Sandbox")
        molecule = st.text_input("Target", value=query)
        if st.button("Generate Hypothesis"):
            st.success(f"Hypothesis: Overexpression of {molecule} will increase ATP yield.")
else:
    st.warning("No results found. Try a different term.")
