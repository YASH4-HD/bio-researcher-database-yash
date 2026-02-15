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
        df = pd.read_csv("lehninger_index.csv")
        df['text_content'] = df['text_content'].astype(str).str.lower()
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

df = load_index()

# --- 3. PUBMED FUNCTION ---
def search_pubmed(search_query):
    try:
        # Search
        h_search = Entrez.esearch(db="pubmed", term=search_query, retmax=5)
        res_search = Entrez.read(h_search)
        h_search.close()
        
        ids = res_search.get("IdList", [])
        if not ids: return None
            
        # Summary
        h_summ = Entrez.esummary(db="pubmed", id=",".join(ids))
        summaries = Entrez.read(h_summ)
        h_summ.close()
        return summaries
    except Exception as e:
        st.error(f"PubMed API Error: {e}")
        return None

# --- 4. UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.caption("Researcher: Yashwant Nama | Molecular Biology & Computational Research")

# --- 5. SEARCH INPUT (Defined at Top Level to avoid NameError) ---
query = st.sidebar.text_input("Enter Biological Term", value="Glycolysis").lower()

# --- 6. MAIN LOGIC ---
if df is not None and query:
    # Filter CSV for the query
    results = df[df['text_content'].str.contains(query, na=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        page_list = results['page'].tolist()
        selected_page = st.sidebar.selectbox("Select Page to View", page_list)
        
        # Display Images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Isolated Diagram")
            st.image(f"{R2_URL}/diagrams/diag_{selected_page}.png", use_container_width=True)
            st.caption(f"Figure extraction - Page {selected_page}")

        with col2:
            st.subheader("📄 Full Page Context")
            st.image(f"{R2_URL}/full_pages/page_{selected_page}.png", use_container_width=True)
            st.caption(f"Full text context - Page {selected_page}")
                
        # PubMed Section
        st.divider()
        st.subheader(f"📚 Latest Research for '{query.capitalize()}'")
        
        # We use 'query' here safely because it's defined at the top
        if st.button(f"Fetch PubMed Articles"):
            with st.spinner("Searching NCBI Databases..."):
                data = search_pubmed(query)
                if data:
                    for doc in data:
                        title = doc.get('Title', 'No Title')
                        date = doc.get('PubDate', 'N/A')
                        pmid = doc.get('Id', '')
                        st.markdown(f"**{title}**")
                        st.write(f"📅 {date} | [View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        st.write("---")
                else:
                    st.warning("No articles found.")

        with st.expander("View Page OCR Text"):
            text = results[results['page'] == selected_page]['text_content'].values[0]
            st.write(text)
    else:
        st.warning(f"No matches found for '{query}' in the textbook index.")
else:
    st.info("Please enter a search term in the sidebar.")
