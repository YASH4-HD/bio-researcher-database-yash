import streamlit as st
import pandas as pd
from Bio import Entrez
import io

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")
R2_URL = "https://pub-ac8710154cab4570a9ac4ba3d21143e8.r2.dev"
Entrez.email = "yashwant.nama@example.com" 

# --- 2. LOAD DATA (With Fallback) ---
@st.cache_data
def load_index():
    try:
        # Attempt to load your real CSV
        df = pd.read_csv("lehninger_index.csv")
        if df.empty:
            raise ValueError("CSV is empty")
        df['text_content'] = df['text_content'].astype(str).str.lower()
        return df
    except Exception as e:
        st.sidebar.warning(f"Using Demo Mode (CSV Error: {e})")
        # FALLBACK: Create a small demo dataset so the app doesn't crash
        data = {
            'page': [44, 999, 1050],
            'text_content': ['table of contents glycolysis', 'atp synthesis mitochondria', 'insulin signaling pathway']
        }
        return pd.DataFrame(data)

df = load_index()

# --- 3. PUBMED FUNCTION ---
def search_pubmed(search_query):
    try:
        h_search = Entrez.esearch(db="pubmed", term=search_query, retmax=5)
        res_search = Entrez.read(h_search)
        h_search.close()
        
        ids = res_search.get("IdList", [])
        if not ids: return None
            
        h_summ = Entrez.esummary(db="pubmed", id=",".join(ids))
        summaries = Entrez.read(h_summ)
        h_summ.close()
        return summaries
    except Exception as e:
        return None

# --- 4. UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.caption("Researcher: Yashwant Nama | Molecular Biology & Computational Research")

# --- 5. SEARCH INPUT ---
query = st.sidebar.text_input("Enter Biological Term", value="Glycolysis").lower()

# --- 6. MAIN LOGIC ---
if df is not None and query:
    # Filter CSV
    results = df[df['text_content'].str.contains(query, na=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        page_list = results['page'].tolist()
        selected_page = st.sidebar.selectbox("Select Page to View", page_list)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Isolated Diagram")
            st.image(f"{R2_URL}/diagrams/diag_{selected_page}.png", use_container_width=True)

        with col2:
            st.subheader("📄 Full Page Context")
            st.image(f"{R2_URL}/full_pages/page_{selected_page}.png", use_container_width=True)
                
        st.divider()
        st.subheader(f"📚 Latest Research for '{query.capitalize()}'")
        
        if st.button("Fetch PubMed Articles"):
            with st.spinner("Searching NCBI..."):
                data = search_pubmed(query)
                if data:
                    for doc in data:
                        st.markdown(f"**{doc.get('Title', 'No Title')}**")
                        st.write(f"📅 {doc.get('PubDate', 'N/A')} | [View Paper](https://pubmed.ncbi.nlm.nih.gov/{doc.get('Id', '')}/)")
                        st.write("---")
                else:
                    st.warning("No articles found.")

        with st.expander("View Page OCR Text"):
            text = results[results['page'] == selected_page]['text_content'].values[0]
            st.write(text)
    else:
        st.warning(f"No matches found for '{query}'.")
else:
    st.info("Please enter a search term in the sidebar.")
