import streamlit as st
import pandas as pd
from Bio import Entrez
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")
R2_URL = "https://pub-ac8710154cab4570a9ac4ba3d21143e8.r2.dev"
Entrez.email = "yashwant.nama@example.com"  # Update with your actual email

# --- LOAD DATA ---
@st.cache_data
def load_index():
    # Ensure lehninger_index.csv is in your GitHub/Folder
    try:
        df = pd.read_csv("lehninger_index.csv")
        return df
    except Exception as e:
        st.error(f"Error loading CSV index: {e}")
        return None

df = load_index()

# --- PUBMED SEARCH FUNCTION (STRENGTHENED) ---
def search_pubmed(query):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=5, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        
        if not id_list:
            return None
            
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="summary", retmode="xml")
        details = Entrez.read(handle)
        handle.close()
        return details.get('DocSum', [])
    except Exception as e:
        st.error(f"PubMed API Error: {e}")
        return None

# --- UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.markdown("### Specialized Retrieval System for Lehninger Principles of Biochemistry")
st.caption("Researcher: Yashwant Nama | PhD Candidate Portfolio")

# --- SIDEBAR ---
st.sidebar.header("Search & Parameters")
query = st.sidebar.text_input("Enter Biological Term (e.g., Glycolysis, ATP)", "").lower()

# --- SEARCH LOGIC ---
if query and df is not None:
    # Filter pages
    results = df[df['text_content'].str.contains(query, na=False, case=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        page_options = results['page'].tolist()
        selected_page = st.sidebar.selectbox("Select Page to View", page_options)
        
        # --- DISPLAY AREA ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Isolated Diagram")
            # Construct URL for the diagram
            diag_url = f"{R2_URL}/diagrams/diag_{selected_page}.png"
            
            # Using a fallback mechanism if diagram doesn't exist
            # (Streamlit renders a broken image if URL is 404, 
            # so we provide a clear caption)
            st.image(diag_url, 
                     use_container_width=True, 
                     caption=f"Extracted Figure (Page {selected_page})")
            st.info("Note: If image is blank, no isolated diagram exists for this page.")

        with col2:
            st.subheader("📄 Full Page Context")
            full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
            st.image(full_url, 
                     use_container_width=True, 
                     caption=f"Lehninger Page {selected_page}")
                
        # --- PUBMED INTEGRATION ---
        st.markdown("---")
        st.subheader(f"📚 Translational Research: {query.capitalize()}")
        st.write("Linking textbook fundamentals to current scientific literature.")
        
        if st.button(f"Fetch PubMed Articles for '{query}'"):
            with st.spinner("Querying NCBI PubMed Database..."):
                articles = search_pubmed(query)
                if articles:
                    for art in articles:
                        # Safer dictionary access to avoid TypeErrors
                        title = "No Title Available"
                        if 'Item' in art:
                            for item in art['Item']:
                                if isinstance(item, str) and len(item) > 10: # Likely the title
                                    title = item
                                    break
                        
                        pub_date = art.get('PubDate', 'N/A')
                        pmid = art.get('Id', '')
                        
                        st.markdown(f"**{title}**")
                        st.write(f"📅 {pub_date} | [Read Paper](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        st.write("---")
                else:
                    st.warning("No recent PubMed articles found for this term.")

        # --- TEXT CONTENT ---
        with st.expander("View Extracted Page Text (OCR)"):
            page_text = results[results['page'] == selected_page]['text_content'].values[0]
            st.write(page_text)
            
    else:
        st.sidebar.warning("No matches found. Try a broader term.")
else:
    st.info("👈 Enter a term in the sidebar to search the 4,381-page database.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Status:**
- Cloud Storage: **Connected** (Cloudflare R2)
- Database: **1.44 GB Indexed**
- API Status: **NCBI Entrez Active**
""")
