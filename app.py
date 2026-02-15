import streamlit as st
import pandas as pd
from Bio import Entrez
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")
R2_URL = "https://pub-ac8710154cab4570a9ac4ba3d21143e8.r2.dev"
Entrez.email = "your.email@example.com"  # Required by NCBI

# --- LOAD DATA ---
@st.cache_data
def load_index():
    # Ensure lehninger_index.csv is in the same folder as this script
    df = pd.read_csv("lehninger_index.csv")
    return df

try:
    df = load_index()
except Exception as e:
    st.error("Index file not found. Please ensure lehninger_index.csv is present.")
    st.stop()

# --- PUBMED SEARCH FUNCTION ---
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
        return details['DocSum']
    except Exception as e:
        return f"Error: {e}"

# --- UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.markdown("### Cloud-Native Retrieval System for Lehninger Biochemistry")
st.caption("Researcher: Yashwant Nama | Jaipur, Rajasthan")

# --- SIDEBAR SEARCH ---
st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Enter Biological Term (e.g., Glycolysis, ATP)", "").lower()

# --- SEARCH LOGIC ---
if query:
    results = df[df['text_content'].str.contains(query, na=False, case=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        page_options = results['page'].tolist()
        selected_page = st.sidebar.selectbox("Select Page to View", page_options)
        
        # --- DISPLAY AREA ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"🖼️ Isolated Diagram (Page {selected_page})")
            # Using Cloudflare R2 URL instead of local H: drive
            diag_url = f"{R2_URL}/diagrams/diag_{selected_page}.png"
            st.image(diag_url, use_container_width=True, caption=f"Diagram {selected_page}")

        with col2:
            st.subheader(f"📄 Full Page Context")
            # Using Cloudflare R2 URL
            full_url = f"{R2_URL}/full_pages/page_{selected_page}.png"
            st.image(full_url, use_container_width=True, caption=f"Page {selected_page}")
                
        # --- PUBMED INTEGRATION SECTION ---
        st.markdown("---")
        st.subheader(f"📚 Latest Research: {query.capitalize()}")
        
        if st.button(f"Fetch PubMed Articles for '{query}'"):
            with st.spinner("Querying NCBI PubMed Database..."):
                articles = search_pubmed(query)
                if articles:
                    for art in articles:
                        title = art['Item'][0] # Title is usually the first item
                        pub_date = art['PubDate']
                        pmid = art['Id']
                        st.markdown(f"**{title}**")
                        st.write(f"Published: {pub_date} | [View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        st.write("---")
                else:
                    st.warning("No recent PubMed articles found for this term.")

        with st.expander("Show Page Text"):
            page_text = results[results['page'] == selected_page]['text_content'].values[0]
            st.write(page_text)
            
    else:
        st.sidebar.warning("No matches found.")
else:
    st.info("👈 Enter a term in the sidebar to search through the cloud-hosted database.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("System Status: Cloud-Connected (R2)")
