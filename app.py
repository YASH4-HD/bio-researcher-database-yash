import re
from collections import Counter
from itertools import combinations

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
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "have",
    "were",
    "been",
    "about",
    "their",
}


# --- 2. LOAD DATA ---
@st.cache_data
def load_index():
    try:
        df = pd.read_csv("lehninger_index.csv", on_bad_lines="skip")
        if df.empty or len(df.columns) < 2:
            raise ValueError("CSV is empty or poorly formatted")

        df.columns = [c.strip().lower() for c in df.columns]
        if "text_content" in df.columns:
            df["text_content"] = df["text_content"].astype(str).str.lower()
        return df
    except Exception as e:
        st.sidebar.error(f"CSV Error: {e}")
        return pd.DataFrame(
            {
                "page": [44],
                "text_content": [
                    "glycolysis content brief preface foundations metabolism energy coupling"
                ],
            }
        )


@st.cache_data
def generate_query_suggestions(search_query: str, text_series: pd.Series):
    """Generate lightweight semantic suggestions from synonyms + corpus frequency."""
    suggestions = set(BIO_SYNONYMS.get(search_query, []))

    token_counts = Counter()
    for txt in text_series.dropna().head(200):
        tokens = re.findall(r"[a-z]{4,}", str(txt))
        token_counts.update(t for t in tokens if t not in STOP_WORDS)

    for term, _ in token_counts.most_common(30):
        if search_query in term or term in search_query:
            continue
        suggestions.add(term)
        if len(suggestions) >= 8:
            break

    return sorted(suggestions)


@st.cache_data
def compute_concept_connections(results_df: pd.DataFrame, max_rows: int = 120):
    """Build simple co-occurrence edges to surface related concepts."""
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
df = load_index()
st.title("🧬 BioVisual Search Engine")
st.caption("Researcher: Yashwant Nama | Molecular Biology & Computational Research")

# --- 5. SEARCH INPUT ---
query = st.sidebar.text_input("Enter Biological Term", value="Glycolysis").lower().strip()
feature_flags = st.sidebar.multiselect(
    "Explore Unique Feature Additions",
    [
        "Semantic Query Expansion",
        "Concept Co-occurrence Map",
        "Reading List Builder",
    ],
    default=["Semantic Query Expansion", "Concept Co-occurrence Map"],
)

# --- 6. MAIN LOGIC ---
if df is not None and query:
    if "text_content" in df.columns:
        results = df[df["text_content"].str.contains(query, na=False)]
    else:
        results = df

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

            col_img, _ = st.columns([2, 1])
            with col_img:
                full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
                st.image(
                    full_url,
                    caption=f"Lehninger Page {selected_page} (Click to expand)",
                    width=700,
                )

        with tab2:
            st.subheader("Discovery Lab: Experimental Search Features")

            if "Semantic Query Expansion" in feature_flags:
                st.markdown("#### 🔎 Semantic Query Expansion")
                suggestions = generate_query_suggestions(query, df.get("text_content", pd.Series()))
                if suggestions:
                    st.caption("Try one of these related concepts:")
                    st.write(" • " + " • ".join(suggestions[:8]))
                else:
                    st.info("No related suggestions found yet.")

            if "Concept Co-occurrence Map" in feature_flags:
                st.markdown("#### 🕸️ Concept Co-occurrence Map")
                concepts_df = compute_concept_connections(results)
                if concepts_df.empty:
                    st.info("Not enough context to compute concept links.")
                else:
                    st.dataframe(concepts_df, width="stretch", hide_index=True)
                    st.bar_chart(
                        concepts_df.set_index(
                            concepts_df["term_a"] + " ↔ " + concepts_df["term_b"]
                        )["co_occurrences"]
                    )

        with tab3:
            st.subheader(f"Latest Research for '{query.capitalize()}'")
            if st.button("Fetch PubMed Articles"):
                with st.spinner("Searching NCBI..."):
                    data = search_pubmed(query)
                    if data:
                        selected_papers = []
                        for i, doc in enumerate(data):
                            title = doc.get("Title", "No Title")
                            source = doc.get("Source", "Journal")
                            pubdate = doc.get("PubDate", "N/A")
                            pmid = doc.get("Id", "")

                            st.markdown(f"#### {title}")
                            st.write(f"📖 *{source}* | 📅 {pubdate}")
                            st.link_button("Read Full Paper", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

                            if "Reading List Builder" in feature_flags:
                                include = st.checkbox(
                                    "Add to reading list",
                                    key=f"reading_list_{i}_{pmid}",
                                )
                                if include:
                                    selected_papers.append(f"- {title} ({source}, {pubdate})")

                            st.write("---")

                        if "Reading List Builder" in feature_flags and selected_papers:
                            st.markdown("### 🗂️ Your Quick Reading List")
                            st.code("\n".join(selected_papers), language="markdown")
                    else:
                        st.warning("No articles found.")
    else:
        st.warning(f"No matches found for '{query}'.")
