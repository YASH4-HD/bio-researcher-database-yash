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
        if df.empty:
            raise ValueError("CSV is empty")
        df['text_content'] = df['text_content'].astype(str).str.lower()
        return df
    except Exception as e:
        st.sidebar.warning(f"Demo Mode Active (CSV Error: {e})")
        data = {'page': [44, 999], 'text_content': ['glycolysis pathway', 'atp synthesis']}
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
    results = df[df['text_content'].str.contains(query, na=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        selected_page = st.sidebar.selectbox("Select Page to View", results['page'].tolist())
        
        # --- NEW SINGLE COLUMN LAYOUT ---
        st.subheader(f"📄 Textbook Context: Page {selected_page}")
        
        # Display full page image. 
        # Streamlit automatically allows "Click to enlarge" on images.
        full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
        st.image(full_url, 
                 caption="Click image to view full size", 
                 use_container_width=True)
                
        # --- PUBMED SECTION (UPDATED) ---
        st.divider()
        st.subheader(f"📚 Latest Research for '{query.capitalize()}'")
        
        if st.button("Fetch PubMed Articles"):
            with st.spinner("Searching NCBI..."):
                data = search_pubmed(query)
                if data:
                    for doc in data:
                        # Professional formatting
                        st.markdown(f"#### {doc.get('Title', 'No Title')}")
                        journal = doc.get('Source', 'Unknown Journal')
                        date = doc.get('PubDate', 'N/A')
                        st.write(f"📖 *{journal}* | 📅 {date}")
                        
                        # Added Link Button
                        st.link_button("Read Full Paper on PubMed", 
                                       f"https://pubmed.ncbi.nlm.nih.gov/{doc.get('Id', '')}/")
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
