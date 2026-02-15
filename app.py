import streamlit as st
import pandas as pd
from Bio import Entrez
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")
R2_URL = "https://pub-ac8710154cab4570a9ac4ba3d21143e8.r2.dev"
Entrez.email = "yashwant.nama@example.com" 

# --- LOAD DATA ---
@st.cache_data
def load_index():
    try:
        df = pd.read_csv("lehninger_index.csv")
        return df
    except Exception as e:
        return None

df = load_index()

# --- IMPROVED PUBMED FUNCTION ---
def search_pubmed(query):
    try:
        # Search for the term
        search_term = f"{query}[Title/Abstract]" # Focus on Title/Abstract for better results
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=5, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record.get("IdList", [])
        if not id_list:
            return None
            
        # Fetch summaries
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="summary", retmode="xml")
        details = Entrez.read(handle)
        handle.close()
        
        # Return the list of summaries
        return details.get('DocSum', [])
    except Exception as e:
        st.error(f"PubMed Error: {e}")
        return None

# --- UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.markdown("### Specialized Retrieval System for Lehninger Principles of Biochemistry")

# --- SIDEBAR ---
st.sidebar.header("Search & Parameters")
query = st.sidebar.text_input("Enter Biological Term", "Glycolysis").lower()

if query and df is not None:
    results = df[df['text_content'].str.contains(query, na=False, case=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        selected_page = st.sidebar.selectbox("Select Page to View", results['page'].tolist())
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Isolated Diagram")
            diag_url = f"{R2_URL}/diagrams/diag_{selected_page}.png"
            # Use a container to catch errors
            st.image(diag_url, use_container_width=True, caption=f"Figure from Page {selected_page}")
            st.caption("If image is missing, this page contains text/tables only.")

        with col2:
            st.subheader("📄 Full Page Context")
            full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
            st.image(full_url, use_container_width=True, caption=f"Lehninger Page {selected_page}")
                
        # --- PUBMED INTEGRATION ---
        st.markdown("---")
        st.subheader(f"📚 Translational Research: {query.capitalize()}")
        
        if st.button(f"Fetch Latest Research for '{query}'"):
            with st.spinner("Searching PubMed..."):
                articles = search_pubmed(query)
                if articles:
                    for art in articles:
                        # Extract title safely from the DocSum structure
                        title = "Untitled Article"
                        for item in art.get('Item', []):
                            if item.attributes.get('Name') == 'Title':
                                title = item
                        
                        pmid = art.get('Id', '')
                        pub_date = "N/A"
                        for item in art.get('Item', []):
                            if item.attributes.get('Name') == 'PubDate':
                                pub_date = item

                        st.markdown(f"**{title}**")
                        st.write(f"📅 {pub_date} | [Read on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        st.write("---")
                else:
                    st.warning(f"No specific results for '{query}'. Try a broader term like 'Metabolism'.")

        with st.expander("View Extracted Page Text (OCR)"):
            page_text = results[results['page'] == selected_page]['text_content'].values[0]
            st.write(page_text)
