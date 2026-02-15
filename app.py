import streamlit as st
import pandas as pd
from Bio import Entrez

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")
R2_URL = "https://pub-ac8710154cab4570a9ac4ba3d21143e8.r2.dev"
Entrez.email = "yashwant.nama@example.com" 

# --- 2. LOAD DATA ---
@st.cache_data
def load_index():
    try:
        # Check if file exists and is not empty
        df = pd.read_csv("lehninger_index.csv", on_bad_lines='skip')
        if df.empty or len(df.columns) < 2:
            raise ValueError("CSV is empty or poorly formatted")
        
        df.columns = [c.strip().lower() for c in df.columns]
        if 'text_content' in df.columns:
            df['text_content'] = df['text_content'].astype(str).str.lower()
        return df
    except Exception as e:
        st.sidebar.error(f"CSV Error: {e}")
        # Fallback for demonstration
        return pd.DataFrame({'page': [44], 'text_content': ['glycolysis content brief preface foundations']})

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
    except:
        return None

# --- 4. UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.caption("Researcher: Yashwant Nama | Molecular Biology & Computational Research")

# --- 5. SEARCH INPUT ---
query = st.sidebar.text_input("Enter Biological Term", value="Glycolysis").lower()

# --- 6. MAIN LOGIC ---
if df is not None and query:
    # Search logic
    if 'text_content' in df.columns:
        results = df[df['text_content'].str.contains(query, na=False)]
    else:
        results = df # Fallback

    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        selected_page = st.sidebar.selectbox("Select Page to View", results['page'].tolist())
        
        # --- IMPROVED IMAGE DISPLAY ---
        st.subheader(f"📄 Textbook Context: Page {selected_page}")
        
        # We use a container to center the image and limit width
        col_img, _ = st.columns([2, 1]) 
        with col_img:
            full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
            # Setting width=700 prevents it from being "too big"
            st.image(full_url, 
                     caption=f"Lehninger Page {selected_page} (Click to expand)", 
                     width=700) 
                
        # --- PUBMED SECTION ---
        st.divider()
        st.subheader(f"📚 Latest Research for '{query.capitalize()}'")
        
        if st.button("Fetch PubMed Articles"):
            with st.spinner("Searching NCBI..."):
                data = search_pubmed(query)
                if data:
                    for doc in data:
                        st.markdown(f"#### {doc.get('Title', 'No Title')}")
                        st.write(f"📖 *{doc.get('Source', 'Journal')}* | 📅 {doc.get('PubDate', 'N/A')}")
                        st.link_button("Read Full Paper", f"https://pubmed.ncbi.nlm.nih.gov/{doc.get('Id', '')}/")
                        st.write("---")
                else:
                    st.warning("No articles found.")
    else:
        st.warning(f"No matches found for '{query}'.")
