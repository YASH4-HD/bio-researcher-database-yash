import hashlib
import io
import re
import textwrap
from collections import Counter
from itertools import combinations
from pathlib import Path
from urllib.parse import quote_plus

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
    "across", "between", "within", "using", "used", "show", "study", "role", "university",
    "vetbooks", "vetbook", "chapter", "edition", "pages", "copyright", "isbn", "www", "http", "other",
    "some", "many", "these", "those", "very", "also", "such",
}
GENE_BLACKLIST = {
    "DNA", "RNA", "ATP", "NAD", "NADH", "CELL", "HUMAN", "MOUSE", "BRAIN", "COVID",
}
METABOLITE_HINTS = {
    "pyruvate", "lactate", "glucose", "fructose", "succinate", "citrate", "acetyl coa",
    "oxaloacetate", "malate", "ketone", "glycogen", "cholesterol", "fatty acid", "atp", "nadh",
}
ENZYME_HINTS = {"kinase", "dehydrogenase", "synthase", "carboxylase", "mutase", "isomerase", "phosphatase", "polymerase"}
LIKELY_ADJ_SUFFIX = ("al", "ic", "ous", "ive", "ary", "ory", "tional")
LIKELY_NOUN_SUFFIX = ("ase", "osis", "tion", "ment", "ity", "ism", "gen", "ome", "ide")
CLINICAL_NOTES = {
    "pyruvate dehydrogenase": "PDH deficiency can lead to lactic acidosis and neurological dysfunction.",
    "lactate": "Elevated lactate may indicate hypoxia, sepsis, or mitochondrial dysfunction.",
    "ketone": "Ketone body overproduction is seen in diabetic ketoacidosis and fasting states.",
    "cholesterol": "Hypercholesterolemia is strongly linked to atherosclerotic cardiovascular disease.",
}


def normalize_token(token: str):
    token = token.lower().strip()
    protected_suffixes = ("sis", "is", "us", "ss")
    if token.endswith(protected_suffixes):
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith(("ses", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def pseudo_pos_ok(token: str):
    if len(token) < 4 or token in STOP_WORDS:
        return False
    if token.endswith(LIKELY_ADJ_SUFFIX) or token.endswith(LIKELY_NOUN_SUFFIX):
        return True
    if any(k in token for k in ["metabol", "protein", "enzym", "cell", "gene", "pathway", "signal"]):
        return True
    return token.isalpha() and len(token) >= 5


def format_point(text: str):
    clean = text.replace("fig.", "").replace("table", "").replace("  ", " ").strip()
    clean = clean.capitalize()
    return textwrap.fill(clean, width=120)


def highlight_entities(text: str):
    rendered = text
    # highlight enzyme-like tokens in blue
    rendered = re.sub(
        r"\b([A-Za-z-]+(?:ase|kinase|synthase|carboxylase|mutase|isomerase|phosphatase|polymerase))\b",
        r"<span style='color:#1f77b4;font-weight:700'>\1</span>",
        rendered,
        flags=re.I,
    )
    # highlight metabolite-like terms in green
    metabolite_pattern = r"\b(" + "|".join(sorted(re.escape(m) for m in METABOLITE_HINTS if " " not in m)) + r")\b"
    rendered = re.sub(metabolite_pattern, r"<span style='color:#2ca02c;font-weight:700'>\1</span>", rendered, flags=re.I)
    return rendered


def maybe_reaction_equation(text: str):
    if "+" in text and "->" in text:
        left, right = text.split("->", 1)
        return left.replace("+", " + ") + r" \rightarrow " + right.replace("+", " + ")
    return None


def find_clinical_note(text: str):
    low = text.lower()
    for k, v in CLINICAL_NOTES.items():
        if k in low:
            return v
    return "No specific clinical annotation found; consider connecting this pathway to disease context manually."


def render_bar_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x_col], df[y_col], color="#4c78a8")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf


# --- 2. LOAD DATA ---
@st.cache_data
def load_index():
    fallback_df = pd.DataFrame(
        {
            "page": [44],
            "text_content": [
                "glycolysis is a central metabolic pathway for energy generation through enzyme catalyzed reactions"
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
        for token in tokens:
            norm = normalize_token(token)
            if pseudo_pos_ok(norm):
                token_counts.update([norm])

    for term, _ in token_counts.most_common(35):
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
        tokens = {normalize_token(t) for t in tokens if pseudo_pos_ok(normalize_token(t))}
        for a, b in combinations(sorted(tokens), 2):
            pair_counts[(a, b)] += 1

    top_pairs = [{"term_a": a, "term_b": b, "co_occurrences": c} for (a, b), c in pair_counts.most_common(20)]
    return pd.DataFrame(top_pairs)


def keyword_set(text: str):
    return {normalize_token(t) for t in re.findall(r"[a-z]{5,}", text.lower()) if pseudo_pos_ok(normalize_token(t))}


def extract_top_study_points(results_df: pd.DataFrame, search_query: str, top_n: int = 10):
    text_blob = " ".join(results_df["text_content"].fillna("").astype(str).head(120).tolist())
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text_blob)

    cleaned = []
    for sent in raw_sentences:
        sent = re.sub(r"\s+", " ", sent).strip(" -•\t")
        if len(sent) < 30:
            continue
        if any(noise in sent for noise in ["copyright", "isbn", "vetbooks", "vetbook"]):
            continue
        cleaned.append(format_point(sent))

    def score(sentence: str):
        low = sentence.lower()
        query_hits = low.count(search_query.lower()) * 3
        bio_hits = sum(low.count(k) for k in ["enzyme", "pathway", "metabolism", "reaction", "regulation", "energy", "cell", "protein", "gene", "substrate"])
        length_bonus = 1 if 60 <= len(sentence) <= 260 else 0
        return query_hits + bio_hits + length_bonus

    ranked = sorted(cleaned, key=score, reverse=True)
    seen, points = set(), []
    for sentence in ranked:
        canonical = sentence.lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        points.append(sentence)
        if len(points) == top_n:
            break
    return points


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


def extract_doi(doc: dict):
    doi = doc.get("DOI", "")
    if isinstance(doi, str) and doi.strip():
        return doi.strip()

    article_ids = doc.get("ArticleIds") or doc.get("ArticleIdList") or []
    if isinstance(article_ids, list):
        for item in article_ids:
            if isinstance(item, dict):
                id_type = str(item.get("IdType", "")).lower()
                value = item.get("Value") or item.get("value")
                if id_type == "doi" and value:
                    return str(value)
            elif isinstance(item, str) and item.lower().startswith("10."):
                return item

    elocation = str(doc.get("ELocationID", ""))
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", elocation, flags=re.I)
    return match.group(0) if match else ""


def is_metabolite_like(search_term: str):
    normalized = search_term.lower().strip()
    return any(m in normalized for m in METABOLITE_HINTS)


def paper_score(doc: dict, search_term: str):
    title = str(doc.get("Title", "")).lower()
    source = str(doc.get("Source", "")).lower()
    pubdate = str(doc.get("PubDate", ""))

    relevance = title.count(search_term.lower()) * 3
    relevance += sum(title.count(k) for k in ["metabolism", "pathway", "enzyme", "gene", "cell"])

    recency = 0
    year_match = re.search(r"(20\d{2})", pubdate)
    if year_match:
        year = int(year_match.group(1))
        recency = max(0, year - 2018)

    prestige = 2 if any(j in source for j in ["nature", "science", "cell", "lancet"]) else 0
    doi_bonus = 1 if extract_doi(doc) else 0
    return relevance + recency + prestige + doi_bonus


def smart_summary_lite(titles: list[str], query: str):
    if not titles:
        return "No paper titles available to summarize."
    top = titles[:3]
    key_terms = Counter()
    for title in top:
        for tok in re.findall(r"[a-z]{5,}", title.lower()):
            norm = normalize_token(tok)
            if pseudo_pos_ok(norm):
                key_terms.update([norm])
    keywords = ", ".join([k for k, _ in key_terms.most_common(5)]) or query
    return (
        f"For **{query}**, the top recent papers emphasize {keywords}. "
        "Across these studies, the trend suggests movement from foundational biochemistry toward targeted and translational applications."
    )


def deterministic_expression(symbol: str):
    tissues = ["Liver", "Muscle", "Brain", "T-cell", "Kidney"]
    digest = hashlib.sha256(symbol.encode()).hexdigest()
    vals = [int(digest[i:i + 2], 16) / 255 for i in range(0, 10, 2)]
    return pd.DataFrame({"tissue": tissues, "expression": vals})


def build_quiz(points: list[str], topic: str):
    if not points:
        return []
    base = points[:3]
    quiz = []
    for i, p in enumerate(base, start=1):
        tokens = [t for t in re.findall(r"[A-Za-z]{5,}", p) if t.lower() not in STOP_WORDS]
        target = tokens[0] if tokens else topic
        quiz.append(
            {
                "q": f"Q{i}: Which term is most central in this point about {topic}?",
                "options": [target, "Mitochondria", "Ribosome", "Apoptosis"],
                "answer": target,
            }
        )
    return quiz


def find_intermediate_nodes(term_a: str, term_b: str, concept_df: pd.DataFrame):
    neighbors = {}
    for _, row in concept_df.iterrows():
        a = row["term_a"]
        b = row["term_b"]
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    set_a = neighbors.get(term_a, set())
    set_b = neighbors.get(term_b, set())
    return sorted(list(set_a & set_b))


# --- 3. PUBMED FUNCTION ---
def search_pubmed(search_query, author_filter=""):
    try:
        term = search_query
        if author_filter.strip():
            term = f"({search_query}) AND ({author_filter}[Author])"

        h_search = Entrez.esearch(db="pubmed", term=term, retmax=12)
        res_search = Entrez.read(h_search)
        h_search.close()

        ids = res_search.get("IdList", [])
        if not ids:
            return None

        h_summ = Entrez.esummary(db="pubmed", id=",".join(ids))
        summaries = Entrez.read(h_summ)
        h_summ.close()
        return list(summaries)
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
lab_mode = st.sidebar.toggle("Lab-Specific Mode")
pi_name = st.sidebar.text_input("PI / Author name", value="") if lab_mode else ""
feature_flags = st.sidebar.multiselect(
    "Explore Unique Feature Additions",
    ["Semantic Query Expansion", "Semantic Bridge", "Visual Knowledge Graph", "Reading List Builder", "Metabolic Map Link"],
    default=["Semantic Query Expansion", "Semantic Bridge", "Visual Knowledge Graph", "Metabolic Map Link"],
)

# --- 6. MAIN LOGIC ---
if df is not None and query:
    results = df[df["text_content"].str.contains(query, na=False)] if "text_content" in df.columns else df

    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 Textbook Context", "🧠 Discovery Lab", "📚 Literature", "🎯 10 Points", "⚖️ Comparison"])

        with tab1:
            selected_page = st.sidebar.selectbox("Select Page to View", results["page"].tolist())
            st.subheader(f"Textbook Context: Page {selected_page}")
            full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
            with st.container(border=True):
                st.markdown("#### ▢ Interactive Textbook View")
                st.image(full_url, caption=f"Lehninger Page {selected_page} (framed preview)", width=520)
            st.caption(f"Snippet: {str(results.iloc[0].get('text_content', ''))[:220]}...")

        with tab2:
            st.subheader("Discovery Lab: Enhanced Exploration")
            if "Semantic Query Expansion" in feature_flags:
                st.markdown("#### 🔎 Semantic Query Expansion")
                suggestions = generate_query_suggestions(query, df.get("text_content", pd.Series(dtype=str)))
                if suggestions:
                    st.write(" • " + " • ".join(suggestions[:10]))
                    st.caption("Filtered for noun/adjective-like biological terms and OCR noise removal.")
                else:
                    st.info("No related suggestions found yet.")

            concepts_df = compute_concept_connections(results)
            if "Visual Knowledge Graph" in feature_flags and not concepts_df.empty:
                st.markdown("#### 🕸️ Visual Knowledge Graph")
                dot = "graph G {\nlayout=neato; overlap=false; splines=true;\n"
                dot += f'"{query}" [shape=box, style="filled", fillcolor="#d1ecff", color="#1f77b4", penwidth=2];\n'
                for _, row in concepts_df.head(10).iterrows():
                    dot += f'"{row["term_a"]}" -- "{row["term_b"]}" [label="{int(row["co_occurrences"])}", color="#7aa6d8"];\n'
                dot += "}"
                st.graphviz_chart(dot)
                st.caption("Edge weight = term co-occurrence frequency within indexed textbook content.")

                fig_data = concepts_df.head(10).copy()
                fig_data["edge"] = fig_data["term_a"] + "↔" + fig_data["term_b"]
                fig_buf = render_bar_figure(fig_data, "edge", "co_occurrences", "Knowledge Graph Edge Weights")
                if fig_buf:
                    st.download_button("Download Figure (Knowledge Graph PNG)", data=fig_buf, file_name=f"{query}_knowledge_graph.png", mime="image/png")

            if "Metabolic Map Link" in feature_flags and is_metabolite_like(query):
                st.markdown("#### 🧭 Metabolic Map Shortcut")
                st.link_button("Open KEGG pathway search", f"https://www.kegg.jp/kegg-bin/search_pathway_text?map=&keyword={quote_plus(query)}&mode=1")

            if "Semantic Bridge" in feature_flags:
                st.markdown("#### 🌉 Semantic Analysis: Textbook ↔ Literature Bridge")
                st.caption("Fetch papers in Literature tab, then synthesize a compact summary.")

        with tab3:
            st.subheader(f"Latest Research for '{query.capitalize()}'")
            fetch = st.button("Fetch PubMed Articles")
            if fetch:
                with st.spinner("Searching NCBI..."):
                    data = search_pubmed(query, pi_name)
                    st.session_state["pubmed_docs"] = data if data else []

            docs = st.session_state.get("pubmed_docs", [])
            if docs:
                docs = sorted(docs, key=lambda d: paper_score(d, query), reverse=True)
                selected_papers, paper_titles = [], []
                st.caption("Sorted by Paper Score = relevance + recency + journal prestige + DOI availability.")

                for i, doc in enumerate(docs):
                    title = doc.get("Title", "No Title")
                    source = doc.get("Source", "Journal")
                    pubdate = doc.get("PubDate", "N/A")
                    pmid = doc.get("Id", "")
                    paper_titles.append(title)

                    st.markdown(f"#### {title}")
                    st.write(f"📖 *{source}* | 📅 {pubdate} | ⭐ Score: {paper_score(doc, query)}")

                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.link_button("Read Full Paper", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                    with col_b:
                        doi = extract_doi(doc)
                        if doi:
                            st.link_button("DOI", f"https://doi.org/{doi}")
                        else:
                            st.caption("DOI not available in PubMed summary")

                    symbols = extract_gene_symbols(title)
                    if symbols:
                        cols = st.columns(len(symbols))
                        for j, sym in enumerate(symbols):
                            with cols[j]:
                                st.link_button(f"NCBI {sym}", f"https://www.ncbi.nlm.nih.gov/gene/?term={sym}")
                                st.link_button(f"UniProt {sym}", f"https://www.uniprot.org/uniprotkb?query={quote_plus(sym)}")
                                st.link_button(f"PubChem {sym}", f"https://pubchem.ncbi.nlm.nih.gov/#query={quote_plus(sym)}")
                                with st.expander(f"Expression viewer: {sym}"):
                                    exp_df = deterministic_expression(sym)
                                    st.bar_chart(exp_df.set_index("tissue"))
                                    st.caption("Data source (simulated viewer): inspired by GTEx Portal / Human Protein Atlas API.")
                                    exp_buf = render_bar_figure(exp_df, "tissue", "expression", f"Expression profile: {sym}")
                                    if exp_buf:
                                        st.download_button(
                                            f"Download Figure ({sym} expression PNG)",
                                            data=exp_buf,
                                            file_name=f"{sym}_expression.png",
                                            mime="image/png",
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

                st.markdown("### 🧠 Smart Summary (LLM-Lite)")
                st.info(smart_summary_lite(paper_titles, query))
            elif fetch:
                st.warning("No articles found.")

        with tab4:
            st.subheader(f"10 Key Points for '{query.capitalize()}'")
            study_points = extract_top_study_points(results, query, top_n=10)
            clinical_toggle = st.toggle("Show Clinical Notes", value=True)

            if not study_points:
                st.info("Not enough sentence-level textbook context found for this query yet.")
            else:
                for idx, point in enumerate(study_points, start=1):
                    with st.expander(f"Point {idx}", expanded=(idx <= 2)):
                        highlighted = highlight_entities(point)
                        st.markdown(highlighted, unsafe_allow_html=True)
                        eq = maybe_reaction_equation(point)
                        if eq:
                            st.latex(eq)
                        if clinical_toggle:
                            st.caption("Clinical note: " + find_clinical_note(point))

                st.download_button(
                    label="Download 10 Points (.txt)",
                    data="\n".join([f"{i+1}. {p}" for i, p in enumerate(study_points)]),
                    file_name=f"{query.replace(' ', '_')}_10_points.txt",
                    mime="text/plain",
                )

                if st.button("Test My Knowledge (Generate 3 MCQs)"):
                    quiz = build_quiz(study_points, query)
                    for q in quiz:
                        st.markdown(f"**{q['q']}**")
                        st.write("A)", q["options"][0], " | B)", q["options"][1], " | C)", q["options"][2], " | D)", q["options"][3])
                        st.caption(f"Answer: {q['answer']}")

        with tab5:
            st.subheader("Concept Comparison Tool")
            term_b = st.text_input("Compare with second term", value="hypoxia").lower().strip()
            if term_b:
                results_b = df[df["text_content"].str.contains(term_b, na=False)]
                pages_a, pages_b = set(results["page"].tolist()), set(results_b["page"].tolist())
                overlap_pages = sorted(list(pages_a & pages_b))
                st.write(f"**{query}** pages: {len(pages_a)} | **{term_b}** pages: {len(pages_b)} | overlap: {len(overlap_pages)}")
                if overlap_pages:
                    st.caption("Overlapping pages: " + ", ".join(map(str, overlap_pages[:20])))

                concepts_a_df = compute_concept_connections(results).head(25)
                concepts_b_df = compute_concept_connections(results_b).head(25)
                concepts_a = set(concepts_a_df["term_a"].tolist() + concepts_a_df["term_b"].tolist()) if not concepts_a_df.empty else set()
                concepts_b = set(concepts_b_df["term_a"].tolist() + concepts_b_df["term_b"].tolist()) if not concepts_b_df.empty else set()
                overlap_terms = sorted(list(concepts_a & concepts_b))
                st.markdown("#### Overlapping concept nodes")
                if overlap_terms:
                    st.write(" • " + " • ".join(overlap_terms[:15]))
                else:
                    st.info("No strong concept overlap found yet.")

                if st.button("Find Intermediate Nodes"):
                    union_df = pd.concat([concepts_a_df, concepts_b_df], ignore_index=True)
                    bridge_nodes = find_intermediate_nodes(query, term_b, union_df)
                    if bridge_nodes:
                        st.success("Potential bridge nodes: " + ", ".join(bridge_nodes[:10]))
                    else:
                        st.info("No direct intermediate nodes detected from current concept graph.")
    else:
        st.warning(f"No matches found for '{query}'.")
