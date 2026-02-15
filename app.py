import datetime
import hashlib
import importlib.util
import io
import json
import re
import textwrap
# Add this to your imports at the top
# pip install stmol
from stmol import showmol
import py3Dmol
from datetime import date
from collections import Counter
from itertools import combinations
from pathlib import Path
from urllib import error, request
from urllib.parse import quote_plus
import pytz
from datetime import datetime
import pandas as pd
import streamlit as st
from Bio import Entrez
from Bio.SeqUtils import gc_fraction, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from urllib.parse import quote_plus

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




def fetch_wikipedia_summary(topic: str):
    topic = topic.strip()
    if not topic:
        return ""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(topic)}"
    req = request.Request(url, headers={"User-Agent": "BioVisual-Search/1.0"}, method="GET")
    try:
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("extract", "No summary found.")
    except Exception:
        return "Could not fetch Wikipedia summary right now."


def translate_to_hindi(text: str):
    if not text.strip():
        return ""
    q = quote_plus(text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=hi&dt=t&q={q}"
    req = request.Request(url, headers={"User-Agent": "BioVisual-Search/1.0"}, method="GET")
    try:
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        chunks = data[0] if isinstance(data, list) and data else []
        return "".join(c[0] for c in chunks if isinstance(c, list) and c)
    except Exception:
        return "Translation unavailable at the moment."


def render_sidebar_status():
    today = date.today()
    june_exam = date(today.year, 6, 1)
    gate_exam = date(2027, 2, 1)

# =========================
# SIDEBAR: BIO-VERIFY PANEL
# =========================
with st.sidebar:

    # Title
    st.title("🛡️ Bio-Verify 2026")

     # --- INDIA TIME (IST) ---
    try:
        ist = pytz.timezone("Asia/Kolkata")
        # If you used 'from datetime import datetime', use this:
        today_dt = datetime.now(ist)
        today_date = today_dt.date()
    except Exception:
        # Fallback if pytz fails
        from datetime import date
        today_date = date.today()

    # FIXED: Changed today_auto to today_date and fixed indentation
    st.subheader(f"🗓️ {today_date.strftime('%d %b %Y')}")
    
    st.divider()


    # --- EXAM DATES ---
    EXAMS = {
        "CSIR NET JUNE": date(2026, 6, 1),
        "GATE 2027": date(2027, 2, 2),
    }

    st.subheader("📆 Exam Countdown")

    for exam, exam_date in EXAMS.items():
        days_left = (exam_date - today_date).days

        if days_left > 0:
            st.info(f"**{exam}**: {days_left} days left")
        elif days_left == 0:
            st.warning(f"**{exam}**: Exam Today!")
        else:
            st.error(f"**{exam}**: Exam completed")

    st.divider()

    # --- STATUS BADGES ---
    st.success("✅ Live API Connection: Active")
    st.info("Verified Data Sources: NCBI, Wikipedia, Google")

    st.divider()
 # --- PROFILE CARD (Image 3 Style) ---
    st.markdown("""
        <div style="background-color: #1e468a; padding: 20px; border-radius: 15px; text-align: center; color: white;">
            <h3 style="margin: 0; color: white;">Yashwant Nama</h3>
            <p style="margin: 5px 0; font-size: 0.9rem; opacity: 0.8;">Developer & Researcher</p>
            <p style="font-weight: bold; font-size: 1rem;">Bio-Informatics & Genetics</p>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                <span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 20px; font-size: 0.8rem;">🧬 Genomics</span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 20px; font-size: 0.8rem;">🕸️ Networks</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("### 🔎 Suggested Searches")
    st.sidebar.caption("PCR • CRISPR • Glycolysis • DNA Repair • T-cell Metabolism")


def get_secret_value(secret_name: str) -> str:
    try:
        secret_value = st.secrets.get(secret_name, "")
    except Exception:
        return ""
    return str(secret_value).strip()

def render_bar_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    if plt is None:
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
def load_knowledge_base():
    csv_path = Path("knowledge_base.csv")
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        kb_df = pd.read_csv(csv_path)
        
        # Clean white spaces from headers
        kb_df.columns = [c.strip() for c in kb_df.columns]
        
        # --- FIXED LINE BELOW ---
        # Added .str before .lower() to handle the pandas column correctly
        if "Topic" in kb_df.columns:
            kb_df = kb_df[kb_df['Topic'].astype(str).str.lower() != 'topic']
        
        # Select all 4 relevant columns
        valid_cols = [c for c in ['Topic', 'Theory', 'Explanation', 'Ten_Points'] if c in kb_df.columns]
        kb_df = kb_df[valid_cols]
        
        kb_df = kb_df.dropna(subset=['Topic']).reset_index(drop=True)
        return kb_df
    except Exception as e:
        # This will now show the specific error if something else fails
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()



kb_df = load_knowledge_base()

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
    # Ensure these are never filtered out
    priority_biowords = {"protein", "enzyme", "glucose", "metabolism", "atp", "cell", "gene", "vitamin"}

    # Increase search depth
    subset = results_df.head(max_rows)
    
    for _, row in subset.iterrows():
        text = str(row.get("text_content", "")).lower()
        # Find all words with 4+ letters
        raw_tokens = re.findall(r"[a-z]{4,}", text)
        
        # Normalize and filter
        tokens = set()
        for t in raw_tokens:
            norm = normalize_token(t)
            # If it's a priority word OR passes the POS check, keep it
            if norm in priority_biowords or pseudo_pos_ok(norm):
                tokens.add(norm)
        
        # Create pairs
        if len(tokens) >= 2:
            for a, b in combinations(sorted(tokens), 2):
                pair_counts[(a, b)] += 1

    top_pairs = [{"term_a": a, "term_b": b, "co_occurrences": c} for (a, b), c in pair_counts.most_common(25)]
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

def sanitize_concept_df(concept_df: pd.DataFrame):
    if concept_df.empty:
        return concept_df
    mask = (concept_df["term_a"] != "vetbook") & (concept_df["term_b"] != "vetbook")
    return concept_df[mask].copy()


def validate_api_key(provider: str, api_key: str):
    key = (api_key or "").strip()
    if not key:
        return False, "Please provide an API key."
    if provider == "Groq":
        if not key.startswith("gsk_"):
            return False, "Groq keys should start with `gsk_`."
        if len(key) < 30:
            return False, "Groq key looks too short. Please paste the full key from Groq Console."
    if provider == "OpenAI" and not key.startswith("sk-"):
        return False, "OpenAI keys should start with `sk-`."
    if provider == "Gemini":
        if not key.startswith("AIza"):
            return False, "Gemini API keys usually start with `AIza`."
    return True, ""


def call_gemini_with_fallback(api_key: str, prompt: str, model_name: str):
    base_headers = {"Content-Type": "application/json"}

    requested = (model_name or "gemini-1.5-flash").replace("models/", "").strip()
    candidates = [requested]
    if not requested.endswith("-latest"):
        candidates.append(f"{requested}-latest")
    candidates.extend(["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"])

    # discover available models for this key/project and prioritize matching ones
    try:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        list_req = request.Request(list_url, headers=base_headers, method="GET")
        with request.urlopen(list_req, timeout=30) as resp:
            model_data = json.loads(resp.read().decode("utf-8"))
        for m in model_data.get("models", []):
            name = str(m.get("name", ""))  # e.g., models/gemini-1.5-flash
            methods = m.get("supportedGenerationMethods", [])
            if name.startswith("models/") and "generateContent" in methods:
                short = name.split("models/", 1)[1]
                if short not in candidates:
                    candidates.append(short)
    except Exception:
        pass

    tried = []
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }

    for model in candidates:
        if model in tried:
            continue
        tried.append(model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=base_headers, method="POST")
        try:
            with request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            cands = data.get("candidates", []) if isinstance(data, dict) else []
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                if parts and isinstance(parts[0], dict) and parts[0].get("text"):
                    return parts[0]["text"]
                return str(data)
        except error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            if exc.code == 404:
                continue  # try next model candidate
            if exc.code in (400, 403):
                return (
                    "Gemini request failed (400/403). Verify API key validity, Generative Language API access, and selected model name."
                    + (f"\nServer details: {body[:280]}" if body else "")
                )
            return f"API error ({exc.code}). Check key/model/provider settings." + (f"\nServer details: {body[:280]}" if body else "")
        except Exception as exc:
            return f"AI request failed: {exc}"

    return "Gemini model not found for this API version/project. Try model `gemini-1.5-flash` or `gemini-1.5-flash-latest`, and ensure Generative Language API is enabled for your Google project."


def call_ai_analyst(provider: str, api_key: str, prompt: str, model_name: str):
    headers = {"Content-Type": "application/json"}
    if provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
    elif provider == "Groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        selected_model = model_name or "llama-3.1-8b-instant"
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
    elif provider == "HuggingFace":
        url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 220, "temperature": 0.2}}
    else:  # Gemini
        return call_gemini_with_fallback(api_key, prompt, model_name)

    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if provider in {"OpenAI", "Groq"}:
            return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("generated_text", "No response")
        return str(data)
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        if provider == "Groq" and exc.code == 403:
            if "1010" in body:
                return (
                    "Groq API returned 403 with error code 1010, which typically means an invalid/revoked API key. "
                    "Please regenerate a fresh key in Groq Console and try again."
                    + (f"\nServer details: {body[:280]}" if body else "")
                )
            return (
                "Groq API returned 403 (forbidden). This usually means: invalid/expired key, "
                "model access restrictions, or disabled project billing. "
                "Try another Groq model (e.g., llama-3.1-8b-instant) and verify your key in Groq Console."
                + (f"\nServer details: {body[:280]}" if body else "")
            )
        return f"API error ({exc.code}). Check key/model/provider settings." + (f"\nServer details: {body[:280]}" if body else "")
    except Exception as exc:
        return f"AI request failed: {exc}"



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


def clean_sequence(raw_seq: str) -> str:
    """Removes FASTA headers and whitespace."""
    lines = raw_seq.strip().splitlines()
    if lines and lines[0].startswith(">"):
        actual_sequence = "".join(lines[1:])
    else:
        actual_sequence = "".join(lines)
    return "".join(actual_sequence.upper().split())

def infer_sequence_type(seq: str) -> str:
    """Detects if DNA or Protein."""
    if not seq: return "Unknown"
    dna_chars = set("ACGTUN")
    prot_chars = set("ABCDEFGHIKLMNPQRSTVWXYZ")
    chars = set(seq)
    if chars.issubset(dna_chars): return "DNA/RNA"
    if chars.issubset(prot_chars): return "Protein"
    return "Unknown"
    
    dna_chars = set("ACGTUN")
    # Amino acids including common ones
    prot_chars = set("ABCDEFGHIKLMNPQRSTVWXYZ")
    chars = set(seq)
    
    # If it only contains ACGTUN, it's likely DNA/RNA
    if chars.issubset(dna_chars):
        return "DNA/RNA"
    # If it contains other amino acid letters, it's a Protein
    if chars.issubset(prot_chars):
        return "Protein"
    
    return "Unknown"


def wallace_tm(primer: str) -> int:
    seq = primer.upper()
    at = seq.count("A") + seq.count("T")
    gc = seq.count("G") + seq.count("C")
    return 2 * at + 4 * gc


def gc_percent(seq: str) -> float:
    if not seq:
        return 0.0
    return gc_fraction(seq) * 100


# --- 4. UI HEADER ---
df, load_warning = load_index()
st.title("🧬 BioVisual Search Engine")
st.caption("Researcher: Yashwant Nama | Molecular Biology & Computational Research")
if load_warning:
    st.sidebar.warning(load_warning)

# --- 5. SEARCH INPUT ---
render_sidebar_status()
query = ""
pi_name = ""
feature_flags = []
selected_page = None
with st.sidebar.expander("🔬 Search Workspace", expanded=True):
    query = st.text_input("Enter Biological Term", value="Glycolysis").lower().strip()
    lab_mode = st.toggle("Lab-Specific Mode")
    pi_name = st.text_input("PI / Author name", value="") if lab_mode else ""
    feature_flags = st.multiselect(
        "Explore Unique Feature Additions",
        ["Semantic Query Expansion", "Semantic Bridge", "Visual Knowledge Graph", "Reading List Builder", "Metabolic Map Link"],
        default=["Semantic Query Expansion", "Semantic Bridge", "Visual Knowledge Graph", "Metabolic Map Link"],
    )

    if df is not None and query:
        sidebar_results = df[df["text_content"].str.contains(query, na=False)] if "text_content" in df.columns else df
        if not sidebar_results.empty:
            st.success(f"Found in {len(sidebar_results)} pages")
            selected_page = st.selectbox("Select Page to View", sidebar_results["page"].tolist())

# --- 6. MAIN LOGIC ---
if df is not None and query:
    results = df[df["text_content"].str.contains(query, na=False)] if "text_content" in df.columns else df

    if not results.empty:
        if selected_page is None:
            selected_page = results["page"].iloc[0]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(["📖 Textbook Context", "🧠 Discovery Lab", "📚 Literature", "🎯 Key 10 Points", "⚖️ Comparison", "🤖 AI Analyst", "🌐 Global Intelligence", "🇮🇳 Hindi Explain", "🧬 Bioinformatics", "📘 CSIR-NET/GATE", "🎯 CSIR 10-Points", "🧪 Experimental Zone"])

        with tab1:
            st.subheader(f"Textbook Context: Page {selected_page}")
            full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
            with st.container(border=True):
                st.markdown("#### ▢ Interactive Textbook View")
                st.image(full_url, caption=f"Lehninger Page {selected_page} (framed preview)", width=520)
            st.caption(f"Snippet: {str(results.iloc[0].get('text_content', ''))[:220]}...")

        with tab2:
            st.subheader("🧠 Discovery Lab: Enhanced Exploration")
            
            # 1. Semantic Query Expansion
            if "Semantic Query Expansion" in feature_flags:
                st.markdown("#### 🔎 Semantic Query Expansion")
                suggestions = generate_query_suggestions(query, df.get("text_content", pd.Series(dtype=str)))
                if suggestions:
                    st.write(" • " + " • ".join(suggestions[:10]))
                else:
                    st.info("No related suggestions found yet.")

            # 2. Visual Knowledge Graph (Plotly Version - Guaranteed to Render)
            if "Visual Knowledge Graph" in feature_flags:
                st.markdown("#### 🕸️ Visual Knowledge Graph")
                
                concepts_df = sanitize_concept_df(compute_concept_connections(results))
                
                if concepts_df.empty:
                    st.warning(f"⚠️ No connections found for '{query}'.")
                else:
                    if not HAS_PLOTLY or not HAS_NETWORKX:
                        st.warning("Visual Knowledge Graph requires optional dependencies `plotly` and `networkx`.")
                        st.info("Install with: `pip install plotly networkx`")
                    else:
                        import plotly.graph_objects as go
                        import networkx as nx

                        # Create NetworkX graph
                        G = nx.Graph()
                        for _, row in concepts_df.head(15).iterrows():
                            G.add_edge(row['term_a'], row['term_b'], weight=row['co_occurrences'])
    
                        # Calculate layout positions
                        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
                        # Create Edge Traces
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
    
                        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
                        # Create Node Traces
                        node_x, node_y, node_text, node_color, node_hover_text = [], [], [], [], []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(f"{node}")
                            count_a = concepts_df[concepts_df["term_a"] == node]["co_occurrences"].sum()
                            count_b = concepts_df[concepts_df["term_b"] == node]["co_occurrences"].sum()
                            total = int(count_a + count_b)
                            node_hover_text.append(f"Node: {node}<br>Co-occurrences: {total}")
                            # Color logic
                            if node.lower() == query.lower(): node_color.append('#FFD700') # Gold
                            elif "ase" in node.lower(): node_color.append('#C1E1C1') # Green
                            else: node_color.append('#d1ecff') # Blue
    
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                            hovertext=node_hover_text,
                            marker=dict(showscale=False, color=node_color, size=25, line_width=2),
                            hoverinfo='text'
                        )
    
                        # Create Figure
                        fig = go.Figure(data=[edge_trace, node_trace],
                                     layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0,l=0,r=0,t=0),
                                     height=520,
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)))
                        
                        st.plotly_chart(fig, use_container_width=True)
    
                        # --- DOWNLOAD & DATA SECTION ---
                        fig_data = concepts_df.head(10).copy()
                        fig_data["edge"] = fig_data["term_a"] + " ↔ " + fig_data["term_b"]
                        fig_buf = render_bar_figure(fig_data, "edge", "co_occurrences", f"Weights: {query}")
                        
                        col_dl, col_raw = st.columns([1, 1])
                        with col_dl:
                            if fig_buf:
                                st.download_button("⬇️ Download Analysis PNG", fig_buf, f"{query}_analysis.png", "image/png", use_container_width=True)
                        with col_raw:
                            with st.expander("📊 View Connection Table"):
                                st.dataframe(concepts_df, use_container_width=True)
    
    
    
            # 3. Metabolic Map Link
            if "Metabolic Map Link" in feature_flags and is_metabolite_like(query):
                st.markdown("#### 🧭 Metabolic Map Shortcut")
                st.link_button(f"Open KEGG: {query}", f"https://www.kegg.jp/kegg-bin/search_pathway_text?map=&keyword={quote_plus(query)}&mode=1")


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
                                            key=f"dl_expr_{i}_{j}_{pmid}_{sym}",
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
                        with st.expander("Click to reveal answer"):
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

                concepts_a_df = sanitize_concept_df(compute_concept_connections(results)).head(25)
                concepts_b_df = sanitize_concept_df(compute_concept_connections(results_b)).head(25)
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
        with tab6:
            st.subheader("Bio-Analyst Assistant")
            st.caption("Use API key in-session only. No key is stored by this app. For deployment, keep keys in Streamlit secrets.")

            provider = st.selectbox("Provider", ["Gemini", "Groq", "OpenAI", "HuggingFace"])
            api_key = st.text_input("Enter your API key", type="password", help="Use a Groq/OpenAI/HuggingFace key. This field is masked and only kept in-session.")
            secret_key = get_secret_value("GEMINI_API_KEY") if provider == "Gemini" else ""
            effective_api_key = api_key.strip() or secret_key
            if provider == "Gemini" and secret_key and not api_key.strip():
                st.caption("Using `GEMINI_API_KEY` from Streamlit secrets.")

            if provider == "Gemini":
                model_name = st.selectbox(
                    "Gemini model",
                    ["gemini-1.5-flash", "gemini-1.5-pro"],
                    index=0,
                    help="Use gemini-1.5-flash for fast responses and strong compatibility.",
                )
            elif provider == "Groq":
                model_name = st.selectbox(
                    "Groq model",
                    [
                        "llama-3.1-8b-instant",
                        "llama-3.3-70b-versatile",
                        "llama3-70b-8192",
                        "llama3-8b-8192",
                    ],
                    index=0,
                    help="Try llama-3.1-8b-instant first for best compatibility; 70B models are stronger but may have access limits.",
                )
            else:
                model_name = st.text_input("Model (optional)", value="")

            if "ai_prompt" not in st.session_state:
                st.session_state["ai_prompt"] = (
                    "You are a Molecular Biology Assistant. Based on the 10 points from Lehninger and expression context, "
                    "suggest a hypothesis for a metabolic intervention and one experimental validation step."
                )

            st.markdown("#### Prompt templates")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Summarize clinical significance"):
                    st.session_state["ai_prompt"] = (
                        "Summarize the clinical significance of this pathway, include one disease link and one translational biomarker."
                    )
            with c2:
                if st.button("Suggest a CRISPR guide RNA strategy"):
                    st.session_state["ai_prompt"] = (
                        "Suggest a CRISPR guide RNA strategy for one key gene from this context, including controls and expected readouts."
                    )
            with c3:
                if st.button("Analyze metabolic flux in T-cells"):
                    st.session_state["ai_prompt"] = (
                        "Analyze how this topic may alter metabolic flux in T-cells and propose one perturbation experiment."
                    )

            context_points = extract_top_study_points(results, query, top_n=10)
            docs = st.session_state.get("pubmed_docs", [])
            expression_context = ""
            if docs:
                first_symbols = extract_gene_symbols(docs[0].get("Title", ""))
                if first_symbols:
                    sym = first_symbols[0]
                    exp_df = deterministic_expression(sym)
                    expression_context = f"Expression profile for {sym}: " + ", ".join([f"{r.tissue}:{r.expression:.2f}" for r in exp_df.itertuples()])

            user_prompt = st.text_area("Prompt", key="ai_prompt", height=120)

            col_run, col_test = st.columns([1, 1])
            with col_test:
                if st.button("Test API Connection"):
                    ok, msg = validate_api_key(provider, effective_api_key)
                    if not ok:
                        st.warning(msg)
                    else:
                        with st.spinner("Testing provider connection..."):
                            test_resp = call_ai_analyst(provider, effective_api_key, "Reply with OK only.", model_name)
                        if str(test_resp).strip().lower().startswith("ok"):
                            st.success("Connection looks good ✅")
                        else:
                            st.info(test_resp)

            with col_run:
                if st.button("Run AI Analysis"):
                    ok, msg = validate_api_key(provider, effective_api_key)
                    if not ok:
                        st.warning(msg)
                    else:
                        assembled_prompt = (
                            "You are a Senior Molecular Biology and Biochemistry Researcher.\n\n"
                            + "REFERENCE CONTEXT (from user textbook):\n- "
                            + "\n- ".join(context_points[:10])
                            + "\n\n"
                            + f"USER RESEARCH QUESTION:\n{user_prompt}\n\n"
                            + f"EXPRESSION CONTEXT:\n{expression_context or 'Not available'}\n\n"
                            + "INSTRUCTIONS:\n"
                            + "1. Use the REFERENCE CONTEXT to ground your answer in provided material.\n"
                            + "2. If context is missing details, use full internal scientific knowledge to expand rigorously.\n"
                            + "3. If equations are relevant, include LaTeX-ready forms using $$...$$.\n"
                            + "4. Include sections: Biochemical Mechanism, Mathematical Derivation, and Clinical/Research Application.\n"
                            + f"5. Keep analysis specific to topic: {query}."
                        )
                        with st.spinner("Calling AI provider..."):
                            response = call_ai_analyst(provider, effective_api_key, assembled_prompt, model_name)
                        st.session_state["ai_analysis"] = response
                        if provider == "Groq" and ("403" in str(response) or "1010" in str(response)):
                            st.info("Tip: this usually indicates invalid/revoked key. Regenerate key in Groq Console and retry with `llama-3.1-8b-instant`.")
                        if provider == "Gemini" and ("400/403" in str(response) or "Gemini request failed" in str(response)):
                            st.info("Tip: enable Generative Language API for your Google project and verify Gemini key permissions.")

            c_clear, _ = st.columns([1, 4])
            with c_clear:
                if st.button("Clear Analysis"):
                    st.session_state["ai_analysis"] = ""
            ai_out = st.session_state.get("ai_analysis", "")
            if ai_out:
                with st.expander("View Deep Analysis", expanded=True):
                    st.markdown("### AI Analysis")
                    st.write(ai_out)


        with tab7:
            st.subheader("🌐 Global Bio-Intelligence")
            st.caption("Search results matched for accuracy (Wikipedia + PubMed + ResearchGate shortcut).")

            st.markdown("#### 📚 Quick Wikipedia Summary")
            wiki_topic = st.text_input("Search for any topic (e.g., DNA, MITOSIS, CRISPR)", value=query, key="wiki_topic")
            if st.button("Fetch Wikipedia Summary"):
                st.session_state["wiki_summary"] = fetch_wikipedia_summary(wiki_topic)

            wiki_text = st.session_state.get("wiki_summary", "")
            if wiki_text:
                with st.container(border=True):
                    st.markdown(f"### 📚 Research Snapshot: {wiki_topic.title()}")
                    st.write(wiki_text)
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.info("🔗 Source: Wikipedia")
                    with col_meta2:
                        st.info(f"📊 Complexity: {len(wiki_text.split())} words")
                    with col_meta3:
                        st.info("🗓️ Last Updated: Today")
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        st.link_button("📖 Read Full Article", f"https://en.wikipedia.org/wiki/{quote_plus(wiki_topic)}")
                    with col_w2:
                        st.link_button("🔎 Search ResearchGate", f"https://www.google.com/search?q={quote_plus(wiki_topic + ' biology research gate')}")

            st.divider()
            st.markdown("#### 🔬 Technical Research (NCBI)")
            db = st.selectbox("Select Database", ["pubmed"], key="global_db")
            pubmed_topic = st.text_input("Enter pubmed keyword for technical data", value=query, key="global_pubmed_topic")
            if st.button("Search NCBI"):
                st.session_state["global_pubmed_docs"] = search_pubmed(pubmed_topic) or []

            docs_g = st.session_state.get("global_pubmed_docs", [])
            if docs_g:
                for i, doc in enumerate(docs_g[:5], start=1):
                    title = doc.get("Title", "No Title")
                    src = doc.get("Source", "Journal")
                    date_pub = doc.get("PubDate", "N/A")
                    pmid = doc.get("Id", "")
                    with st.expander(f"{i}. {title}"):
                        st.write(f"📖 *{src}* | 📅 {date_pub}")
                        if pmid:
                            st.link_button("Open PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

        with tab8:
            st.subheader("🇮🇳 IN Hindi Explain")
            st.caption("Paste English biology text and convert to Hindi explanation.")

            ex1, ex2 = st.columns(2)
            with ex1:
                if st.button("Use Glycolysis Example"):
                    st.session_state["hindi_input"] = (
                        "Glycolysis is a metabolic pathway that converts glucose into pyruvate and releases energy as ATP."
                    )
            with ex2:
                if st.button("Use Restriction Enzyme Example"):
                    st.session_state["hindi_input"] = "Enzymes that cut DNA at specific palindromic sequences."

            hindi_input = st.text_area("English text", value=st.session_state.get("hindi_input", ""), height=150, key="hindi_input")
            if st.button("Translate to Hindi"):
                st.session_state["hindi_output"] = translate_to_hindi(hindi_input)

            out_hi = st.session_state.get("hindi_output", "")
            if out_hi:
                with st.container(border=True):
                    st.markdown("### Hindi Output")
                    st.info(out_hi)
                    st.download_button(
                        "Download Hindi Output (.txt)",
                        data=out_hi,
                        file_name="hindi_explanation.txt",
                        mime="text/plain",
                    )

        with tab9:
            st.subheader("🌐 Global Bioinformatics Command Center")
            st.caption("Centralized Research Hub: Analyze sequences, visualize structures, and access primary databases.")

            # Create Sub-tabs inside Tab 9
            bio_sub1, bio_sub2, bio_sub3 = st.tabs(["🧬 Sequence Analysis", "💎 3D Structure", "🔍 Database Portal"])

            with bio_sub1:
                st.markdown("#### 🔬 Protein/DNA Sequence Analyzer")
                seq_input = st.text_area("Paste FASTA sequence", height=150)
                if seq_input:
                    # FIX: Ensure the variable name matches the function name
                    cleaned = clean_sequence(seq_input) 
                    seq_type = infer_sequence_type(cleaned)
                    if seq_type == "Protein":
                        st.success(f"🧬 **Detected Type:** {seq_type}")
                    else:
                        st.info(f"🧬 **Detected Type:** {seq_type}")
                    
                    if seq_type == "Protein":
                        try:
                            # Using Biopython's ProteinAnalysis
                            analysed_seq = ProteinAnalysis(cleaned)
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Molecular Weight", f"{analysed_seq.molecular_weight():.2f} Da")
                            col2.metric("Isoelectric Point", f"{analysed_seq.isoelectric_point():.2f}")
                            col3.metric("Aromaticity", f"{analysed_seq.aromaticity():.2f}")
                            st.markdown("---")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"**Sequence Length:** {len(cleaned)} residues")
                            with c2:
                                 # Simple Amino Acid composition check
                                 hydrophobic = sum(cleaned.count(x) for x in "AILMFVPGW")
                                 st.write(f"**Hydrophobic Residues:** {hydrophobic} ({(hydrophobic/len(cleaned))*100:.1f}%)")
                             # --- END OF NEW CODE ---
                        except Exception as e:
                            st.error(f"Analysis error: {e}")
                    elif seq_type == "DNA/RNA":
                        from Bio.SeqUtils import gc_fraction
                        st.metric("Melting Temp (Tm)", f"{tm_value:.2f} °C")
            with bio_sub2:
                st.markdown("#### 💎 3D Molecular Visualization")
                
                # Add quick-load buttons for common structures
                st.write("Quick Load Samples:")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("Spike RBD (6M0J)"): pdb_id_input = "6M0J"
                with c2:
                    if st.button("Insulin (1TRZ)"): pdb_id_input = "1TRZ"
                with c3:
                    if st.button("Hemoglobin (1A3N)"): pdb_id_input = "1A3N"
                with c4:
                    if st.button("DNA Polymerase (1T7P)"): pdb_id_input = "1T7P"

                pdb_id_input = st.text_input("Enter 4-letter PDB ID", value="6M0J").upper().strip()
                
                if len(pdb_id_input) == 4:
                    with st.spinner(f"Fetching {pdb_id_input} from RCSB..."):
                        with st.container(border=True):
                            view = py3Dmol.view(query=f'pdb:{pdb_id_input}')
                            view.setStyle({'cartoon': {'color': 'spectrum'}})
                            # Add a "stick" style for the active site residues
                            view.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'white'})
                            view.zoomTo()
                            showmol(view, height=500, width=800)
                    st.success(f"Displaying PDB: {pdb_id_input}")
                else:
                    st.warning("Please enter a valid 4-character PDB ID.")


            with bio_sub3:
                st.markdown("#### 🔍 Primary Database Access")
                col_sel, col_search = st.columns([2, 1])
                with col_sel:
                    portal_choice = st.selectbox(
                        "Select Database",
                        options=["RCSB PDB", "UniProt", "NCBI Gene", "NCBI Structure", "AlphaFold DB"]
                    )
                with col_search:
                    # If the user hasn't typed anything specific, use the main query
                    portal_query = st.text_input("ID or Search Term", value=query, key="portal_search_input").strip()

                # --- FIXED RCSB PDB URL LOGIC ---
                # Defensive default to avoid NameError if an unexpected option/state appears.
                target_url = f"https://www.google.com/search?q={quote_plus(portal_query)}"
                if portal_choice == "RCSB PDB":
                    # Check if it looks like a PDB ID (4 characters, alphanumeric)
                    if len(portal_query) == 4 and portal_query.isalnum():
                        target_url = f"https://www.rcsb.org/structure/{portal_query}"
                    else:
                        # RCSB expects a JSON `request` payload in the URL.
                        # For `full_text` service, only `value` is valid (no `operator`).
                        rcsb_request = {
                            "query": {
                                "type": "terminal",
                                "service": "full_text",
                                "parameters": {
                                    "value": portal_query,
                                },
                            },
                            "return_type": "entry",
                        }
                        target_url = f"https://www.rcsb.org/search?request={quote_plus(json.dumps(rcsb_request, separators=(',', ':')))}"
                
                elif portal_choice == "UniProt":
                    target_url = f"https://www.uniprot.org/uniprotkb?query={quote_plus(portal_query)}"
                elif portal_choice == "NCBI Gene":
                    target_url = f"https://www.ncbi.nlm.nih.gov/gene/?term={quote_plus(portal_query)}"
                elif portal_choice == "NCBI Structure":
                    target_url = f"https://www.ncbi.nlm.nih.gov/structure/?term={quote_plus(portal_query)}"
                elif portal_choice == "AlphaFold DB":
                    target_url = f"https://alphafold.ebi.ac.uk/search/text/{quote_plus(portal_query)}"
                else:
                    # Handles any stale/unknown selectbox state without crashing.
                    st.warning("Unknown database selection. Falling back to a web search.")

                st.link_button(f"🚀 Open {portal_choice} in New Tab", target_url, use_container_width=True)
                
                # ONLY show iframe in this sub-tab (Bio Sub 3)
                embeddable_portals = {"RCSB PDB", "UniProt"}
                if portal_choice in embeddable_portals:
                    if portal_query:
                        st.components.v1.iframe(target_url, height=800, scrolling=True)
                else:
                    st.info(f"{portal_choice} does not support embedding. Use the button above.")
            
            st.divider() 
                # --- PLACE THIS PART INSIDE YOUR TAB 10 BLOCK ---
        with tab10:
            st.subheader("📘 CSIR-NET / GATE Planner & Study Reader")
            KB_IMAGES_URL = "https://pub-c99877116ebd42b3b53e7d6779b1bfb6.r2.dev"

            if kb_df.empty:
                st.warning("⚠️ knowledge_base.csv not found or empty.")
            else:
                if 'kb_idx' not in st.session_state:
                    st.session_state.kb_idx = 0

                # 1. Navigation Toolbar
                c1, c2, c3, c4 = st.columns([0.6, 0.8, 0.6, 4])
                with c1:
                    if st.button("⬅ PREV", key="kb_prev", use_container_width=True, disabled=st.session_state.kb_idx == 0):
                        st.session_state.kb_idx -= 1
                        st.rerun()
                with c2:
                    curr = st.session_state.kb_idx + 1
                    total = len(kb_df)
                    st.markdown(f'<div style="border: 1px solid #ddd; border-radius: 5px; padding: 2px; text-align: center; background-color: #f0f2f6; line-height: 1.2;"><p style="margin: 0; font-size: 0.7rem; color: #555;">TOPIC</p><p style="margin: 0; font-weight: bold; font-size: 1rem;">{curr} / {total}</p></div>', unsafe_allow_html=True)
                with c3:
                    if st.button("NEXT ➡", key="kb_next", use_container_width=True, disabled=st.session_state.kb_idx == len(kb_df) - 1):
                        st.session_state.kb_idx += 1
                        st.rerun()

                st.divider()
                kb_row = kb_df.iloc[st.session_state.kb_idx]
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    # 1. TOPIC (Column: Topic)
                    topic_name = str(kb_row.get("Topic", "Untitled Topic")).strip()
                    st.header(topic_name)
                    
                    # 2. MAIN THEORY (Column: Theory)
                    st.markdown("### 📖 Theory & Mechanism")
                    theory_text = str(kb_row.get("Theory", "No theory available."))
                    st.write(theory_text)
                    
                    # 3. DETAILED EXPANDER (Shows ONLY the Detailed Explanation)
                    with st.expander("📘 Detailed Analysis & Key Points", expanded=False):
                        st.markdown("**Detailed Breakdown:**")
                        detailed_expl = str(kb_row.get("Explanation", "No detailed explanation available."))
                        st.write(detailed_expl)
                        # Note: We removed the Divider and the Ten_Points section from here 
                        # to avoid redundancy with the CSIR 10-Points tab.

                    # 4. ADD TO REPORT BUTTON
                    if st.button("Add to Research Report", icon="➕", key="kb_report_btn"):
                        if 'report_list' not in st.session_state:
                            st.session_state['report_list'] = []
                        
                        if topic_name not in [item['Topic'] for item in st.session_state['report_list']]:
                            st.session_state['report_list'].append({
                                "Topic": topic_name, 
                                "Notes": detailed_expl # Saving the detailed explanation to the report
                            })
                            st.toast(f"Added {topic_name} to report!", icon="✅")
                        else:
                            st.warning("Topic already in report.")

                with col_right:
                    with st.container(border=True):
                        st.markdown("**🖼️ Topic Diagram**")
                        img_url = f"{KB_IMAGES_URL}/{st.session_state.kb_idx + 1}.jpg"
                        st.image(img_url, use_container_width=True)
                        st.link_button("🔍 Full Image", img_url, use_container_width=True)
        with tab11:
            st.header("🧠 CSIR-NET / GATE: 10 Key Exam Points")
            
            if kb_df.empty:
                st.warning("Please upload knowledge_base.csv")
            else:
                current_kb_idx = st.session_state.get('kb_idx', 0)
                
                if current_kb_idx < len(kb_df):
                    kb_row = kb_df.iloc[current_kb_idx]
                    
                    st.info(f"**Current Topic:** {kb_row.get('Topic', 'N/A')}")
                    
                    # --- DATA CLEANING STEP ---
                    raw_points = kb_row.get("Ten_Points", "")
                    
                    if pd.isna(raw_points) or str(raw_points).strip().lower() == "nan":
                        points_text = "⚠️ No summary found. Check if the 'Ten_Points' column is filled in your CSV."
                    else:
                        # Convert [Alt+Enter] or \n into actual line breaks for Streamlit
                        points_text = str(raw_points).replace('[Alt+Enter]', '\n').replace('_x000D_', '\n')

                    study_mode = st.toggle("Enable Study Mode", key="kb_study_toggle")
                    
                    if study_mode:
                        st.warning("🙈 **Recall the points before revealing!**")
                        if st.button("👁️ Reveal Notes"):
                            st.info(points_text)
                    else:
                        st.success("📝 **Exam Notes:**")
                        st.markdown(points_text)


                    st.divider()
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("📑 Generate Citation", key="kb_cite_btn"):
                            st.code(f"Nama, Y. (2024). {kb_row.get('Topic', 'Topic')}. BioVisual Knowledge Base.")
                    
                    with c2:
                        st.download_button(
                            label="📥 Download Study Notes",
                            data=points_text,
                            file_name=f"{kb_row.get('Topic', 'Notes')}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="kb_dl_btn"
                        )


        
        with tab12:
            st.subheader("🧪 Experimental Zone")
            st.caption("Prototype sandbox for rapid hypothesis testing and molecular design.")
            
            # Create sub-tabs for better organization
            exp_sub1, exp_sub2 = st.tabs(["📋 Experiment Planner", "🧬 Primer Designer"])
            
            with exp_sub1:
                molecule = st.text_input("Target gene/protein/metabolite", value=query, key="exp_target")
                intervention = st.selectbox("Intervention", ["Knockdown", "Overexpression", "Inhibitor", "CRISPR edit"], key="exp_intervention")
                model_system = st.selectbox("Model system", ["T-cells", "HEK293", "Yeast", "Mouse"], key="exp_model")
                
                if st.button("Generate Experiment Draft"):
                    st.session_state["exp_plan"] = (
                        f"**Hypothesis:** Perturbing **{molecule}** via **{intervention}** in **{model_system}** will alter pathway dynamics.\n\n"
                        "**Readouts:** qPCR for transcript levels, Western Blot for protein, and metabolic flux proxy (Lactate/ATP ratio).\n\n"
                        "**Controls:** Non-targeting control (NTC) + Vehicle-only treatment + Baseline wild-type condition."
                    )
                
                if st.session_state.get("exp_plan"):
                    st.success(st.session_state["exp_plan"])

            with exp_sub2:
                st.markdown("#### 🧬 Rapid Primer Property Checker")
                st.caption("Calculate Tm and GC content for PCR primers.")
                primer_seq = st.text_input("Enter Primer Sequence (5' -> 3')", value="GATCGATCGATCGATC", key="primer_input").upper().strip()
                
                if primer_seq:
                    # Count bases
                    at_count = primer_seq.count('A') + primer_seq.count('T')
                    gc_count = primer_seq.count('G') + primer_seq.count('C')
                    seq_len = len(primer_seq)
                    
                    # Professional Tm Calculation Logic
                    if seq_len < 14:
                        # Wallace Rule for very short primers
                        tm = (2 * at_count) + (4 * gc_count)
                    else:
                        # Salt-adjusted formula for longer primers (standard for 14-60 bp)
                        # Formula: 64.9 + 41 * (yGC - 16.4) / n
                        tm = 64.9 + 41 * (gc_count - 16.4) / seq_len
                    
                    gc_content = (gc_count / seq_len) * 100 if seq_len > 0 else 0
                    
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.metric("Melting Temp (Tm)", f"{tm:.1f}°C")
                    p_col2.metric("GC Content", f"{gc_content:.1f}%")
                    p_col3.metric("Length", f"{seq_len} bp")
                    
                    # Updated Scientific Validation Logic
                    if 55 <= tm <= 72:
                        st.success(f"✅ Tm ({tm:.1f}°C) is optimal for high-fidelity PCR.")
                    elif tm > 72:
                        st.warning("⚠️ High Tm: Consider a higher annealing temperature or DMSO.")
                    else:
                        st.warning("⚠️ Low Tm: Consider extending the primer length.")

                   
        # --- END OF ALL TABS ---
        
    else:
        # This ELSE must be indented much less (aligned with 'if not results.empty:')
        st.warning(f"No matches found for '{query}'.")

                        
# --- END OF FILE ---
# =========================
# SIDEBAR: RESEARCH REPORT
# =========================
with st.sidebar:
    st.divider()
    st.markdown("### 📋 My Research Report")
    
        # 1. Initialize the list if it doesn't exist
    if 'report_list' not in st.session_state:
        st.session_state['report_list'] = []

    # 2. Check if the report is empty
    if not st.session_state['report_list']:
        st.info("Your report is empty. Add topics from the 'Reader' tab.")
    else:
        # 3. Show each item in the report
        for i, item in enumerate(st.session_state['report_list']):
            st.write(f"{i+1}. **{item['Topic']}**")
        
        st.write("") # Spacer

        # 4. CLEAR REPORT BUTTON
        if st.button("🗑️ Clear Report", key="clear_sidebar_report", use_container_width=True):
            st.session_state['report_list'] = []
            st.rerun()

        # 5. DOWNLOAD BUTTON LOGIC
        report_text = "RESEARCH REPORT\n" + "="*20 + "\n\n"
        for i, item in enumerate(st.session_state['report_list']):
            report_text += f"{i+1}. TOPIC: {item['Topic']}\n"
            report_text += f"NOTES: {item['Notes']}\n"
            report_text += "-"*20 + "\n"

        st.download_button(
            label="📥 Download Full Report",
            data=report_text,
            file_name="research_report.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_sidebar_report"
        )

    # 6. Research Tip (Moved outside the else so it always shows)
    st.sidebar.markdown("### 💡 Research Tip")
    st.sidebar.info("Focus on molecular interactions and regulatory nodes relevant to CSIR-NET Part C.")
    # --- SIDEBAR RESEARCH LOG ---
    with st.sidebar:
        st.divider()
        st.subheader("📝 Export Research")

        # Formatting decimals safely
        tm_display = f"{tm:.2f}" if ('tm' in locals() and isinstance(tm, (int, float))) else "N/A"
        gc_display = f"{gc_content:.2f}" if ('gc_content' in locals() and isinstance(gc_content, (int, float))) else "N/A"

        # Creating the report string
        summary_text = f"""# 🧬 BioVisual Research Report
**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔍 Target Analysis: {query if 'query' in locals() else 'N/A'}
- **Sequence Type:** {seq_type if 'seq_type' in locals() else 'N/A'}
- **Detected PDB ID:** {pdb_id_input if 'pdb_id_input' in locals() else 'N/A'}

## 🧪 Experimental Design
- **Primer:** `{primer_seq if 'primer_seq' in locals() else 'N/A'}`
- **Melting Temp (Tm):** {tm_display} °C
- **GC Content:** {gc_display}%

## 📝 Experiment Plan
{st.session_state.get('exp_plan', 'No plan generated')}
"""

        # The Download Button
        st.download_button(
            label="Download Research Report",
            data=summary_text,
            file_name=f"Research_Report_{query if 'query' in locals() else 'Export'}.md",
            mime="text/markdown"
        )



