import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
import streamlit as st
from Bio import Entrez

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
    "across", "between", "within", "using", "used", "show", "study", "role",
}
GENE_BLACKLIST = {
    "DNA", "RNA", "ATP", "NAD", "NADH", "CELL", "HUMAN", "MOUSE", "BRAIN", "COVID",
}


# --- 2. LOAD DATA ---
@st.cache_data
def load_index():
    fallback_df = pd.DataFrame(
        {
            "page": [44],
            "text_content": [
                "glycolysis content brief preface foundations metabolism energy coupling enzymes"
            ],
        }
    )

    csv_path = Path("lehninger_index.csv")
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return fallback_df, "Index file is missing/empty. Running in demo mode with sample textbook content."

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        if df.empty or len(df.columns) < 2:
            return fallback_df, "Index file has no usable rows. Running in demo mode."

        df.columns = [c.strip().lower() for c in df.columns]
        if "text_content" not in df.columns or "page" not in df.columns:
            return fallback_df, "Index file is missing required columns ('page', 'text_content'). Running in demo mode."

        df["text_content"] = df["text_content"].astype(str).str.lower()
        return df, None
    except Exception as exc:
        return fallback_df, f"Could not read index ({exc}). Running in demo mode."


@st.cache_data
def generate_query_suggestions(search_query: str, text_series: pd.Series):
    suggestions = set(BIO_SYNONYMS.get(search_query, []))

    token_counts = Counter()
    for txt in text_series.dropna().head(250):
        tokens = re.findall(r"[a-z]{4,}", str(txt))
        token_counts.update(t for t in tokens if t not in STOP_WORDS)

    for term, _ in token_counts.most_common(30):
        if search_query in term or term in search_query:
            continue
        suggestions.add(term)
        if len(suggestions) >= 10:
            break

    return sorted(suggestions)


@st.cache_data
def compute_concept_connections(results_df: pd.DataFrame, max_rows: int = 150):
    pair_counts = Counter()

    for _, row in results_df.head(max_rows).iterrows():
        tokens = set(re.findall(r"[a-z]{5,}", str(row.get("text_content", ""))))
        tokens = {t for t in tokens if t not in STOP_WORDS}
        for a, b in combinations(sorted(tokens), 2):
            pair_counts[(a, b)] += 1

    top_pairs = [
        {"term_a": a, "term_b": b, "co_occurrences": c}
        for (a, b), c in pair_counts.most_common(20)
    ]
    return pd.DataFrame(top_pairs)


def keyword_set(text: str):
    return {
        t for t in re.findall(r"[a-z]{5,}", text.lower())
        if t not in STOP_WORDS
    }


def semantic_bridge_summary(textbook_text: str, article_titles: list[str], query: str):
    textbook_terms = keyword_set(textbook_text)
    pubmed_terms = keyword_set(" ".join(article_titles))
    shared = sorted(list(textbook_terms & pubmed_terms))[:8]

    if not article_titles:
        return None, []

    summary = [
        f"Lehninger context around **{query}** emphasizes foundational mechanisms seen in textbook content.",
        f"Recent papers extend this into research themes like **{', '.join(shared[:4]) if shared else query}**.",
        "Together, this suggests a bridge from core biochemistry concepts to current translational or systems-level studies.",
    ]
    return " ".join(summary), shared


def extract_gene_symbols(text: str):
    candidates = set(re.findall(r"\b[A-Z0-9-]{3,10}\b", text))
    return sorted([c for c in candidates if c not in GENE_BLACKLIST and not c.isdigit()])[:3]


# --- 3. PUBMED FUNCTION ---
def search_pubmed(search_query):
    try:
        h_search = Entrez.esearch(db="pubmed", term=search_query, retmax=8)
        res_search = Entrez.read(h_search)
        h_search.close()

        ids = res_search.get("IdList", [])
        if not ids:
            return None

        h_summ = Entrez.esummary(db="pubmed", id=",".join(ids))
        summaries = Entrez.read(h_summ)
        h_summ.close()
        return summaries
    except Exception:
        return None


# --- 4. UI HEADER ---
df, load_warning = load_index()
st.title("🧬 BioVisual Search Engine")
st.caption("Researcher: Yashwant Nama | Molecular Biology & Computational Research")

if load_warning:
    st.sidebar.warning(load_warning)


# --- 5. SEARCH INPUT ---
query = st.sidebar.text_input("Enter Biological Term", value="Glycolysis").lower().strip()
feature_flags = st.sidebar.multiselect(
    "Explore Unique Feature Additions",
    [
        "Semantic Query Expansion",
        "Semantic Bridge",
        "Visual Knowledge Graph",
        "Reading List Builder",
    ],
    default=["Semantic Query Expansion", "Semantic Bridge", "Visual Knowledge Graph"],
)

# --- 6. MAIN LOGIC ---
if df is not None and query:
    results = df[df["text_content"].str.contains(query, na=False)] if "text_content" in df.columns else df

    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")

        tab1, tab2, tab3 = st.tabs([
            "📖 Textbook Context",
            "🧠 Discovery Lab",
            "📚 Literature",
        ])

        with tab1:
            selected_page = st.sidebar.selectbox("Select Page to View", results["page"].tolist())
            st.subheader(f"Textbook Context: Page {selected_page}")

            full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
            with st.container(border=True):
                st.markdown("#### ▢ Interactive Textbook View")
                st.image(
                    full_url,
                    caption=f"Lehninger Page {selected_page} (framed preview)",
                    width=520,
                )

            snippet = str(results.iloc[0].get("text_content", ""))
            st.caption(f"Snippet: {snippet[:220]}...")

        with tab2:
            st.subheader("Discovery Lab: Enhanced Exploration")

            if "Semantic Query Expansion" in feature_flags:
                st.markdown("#### 🔎 Semantic Query Expansion")
                suggestions = generate_query_suggestions(query, df.get("text_content", pd.Series(dtype=str)))
                if suggestions:
                    st.write(" • " + " • ".join(suggestions[:10]))
                else:
                    st.info("No related suggestions found yet.")

            concepts_df = compute_concept_connections(results)
            if "Visual Knowledge Graph" in feature_flags and not concepts_df.empty:
                st.markdown("#### 🕸️ Visual Knowledge Graph")
                dot = "graph G {\nlayout=neato; overlap=false; splines=true;\n"
                dot += f'"{query}" [shape=box, style="filled", fillcolor="#d1ecff", color="#1f77b4", penwidth=2];\n'
                for _, row in concepts_df.head(10).iterrows():
                    a = row["term_a"]
                    b = row["term_b"]
                    c = int(row["co_occurrences"])
                    dot += f'"{a}" -- "{b}" [label="{c}", color="#7aa6d8"];\n'
                    dot += f'"{query}" -- "{a}" [style=dotted, color="#b2c7e1"];\n'
                dot += "}"
                st.graphviz_chart(dot)

            if "Semantic Bridge" in feature_flags:
                st.markdown("#### 🌉 Semantic Analysis: Textbook ↔ Literature Bridge")
                st.caption("Fetch papers in the Literature tab, then synthesize how textbook foundations connect to current research.")

        with tab3:
            st.subheader(f"Latest Research for '{query.capitalize()}'")
            fetch = st.button("Fetch PubMed Articles")

            if fetch:
                with st.spinner("Searching NCBI..."):
                    data = search_pubmed(query)
                    if data:
                        st.session_state["pubmed_docs"] = data
                    else:
                        st.session_state["pubmed_docs"] = []

            docs = st.session_state.get("pubmed_docs", [])
            if docs:
                selected_papers = []
                paper_titles = []

                for i, doc in enumerate(docs):
                    title = doc.get("Title", "No Title")
                    source = doc.get("Source", "Journal")
                    pubdate = doc.get("PubDate", "N/A")
                    pmid = doc.get("Id", "")
                    paper_titles.append(title)

                    st.markdown(f"#### {title}")
                    st.write(f"📖 *{source}* | 📅 {pubdate}")
                    st.link_button("Read Full Paper", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", key=f"paper_{i}_{pmid}")

                    symbols = extract_gene_symbols(title)
                    if symbols:
                        cols = st.columns(len(symbols))
                        for j, sym in enumerate(symbols):
                            with cols[j]:
                                st.link_button(
                                    f"Gene: {sym}",
                                    f"https://www.ncbi.nlm.nih.gov/gene/?term={sym}",
                                    key=f"gene_{i}_{sym}",
                                )

                    if "Reading List Builder" in feature_flags:
                        include = st.checkbox("Add to reading list", key=f"reading_list_{i}_{pmid}")
                        if include:
                            selected_papers.append(f"- {title} ({source}, {pubdate})")

                    st.write("---")

                if "Reading List Builder" in feature_flags and selected_papers:
                    st.markdown("### 🗂️ Your Quick Reading List")
                    st.code("\n".join(selected_papers), language="markdown")

                if "Semantic Bridge" in feature_flags and st.button("Synthesize Textbook → Research Bridge"):
                    textbook_text = " ".join(results["text_content"].head(3).tolist())
                    bridge_text, shared_terms = semantic_bridge_summary(textbook_text, paper_titles, query)
                    if bridge_text:
                        st.success(bridge_text)
                        if shared_terms:
                            st.caption("Shared key terms: " + ", ".join(shared_terms))
            elif fetch:
                st.warning("No articles found.")
    else:
        st.warning(f"No matches found for '{query}'.")
