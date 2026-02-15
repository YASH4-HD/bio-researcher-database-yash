import streamlit as st
import pandas as pd
from Bio import Entrez

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")
R2_URL = "https://pub-ac8710154cab4570a9ac4ba3d21143e8.r2.dev"
Entrez.email = "yashwant.nama@example.com" 

# --- LOAD DATA WITH ERROR CHECKING ---
@st.cache_data
def load_index():
    try:
        # Check if file exists in the current directory
        df = pd.read_csv("lehninger_index.csv")
        # Ensure text column is string and lowercase for searching
        df['text_content'] = df['text_content'].astype(str).str.lower()
        return df
    except FileNotFoundError:
        st.error("❌ 'lehninger_index.csv' not found! Please ensure it is in the same folder as this script.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

df = load_index()

# --- IMPROVED PUBMED FUNCTION ---
def search_pubmed(query):
    try:
        # 1. Search for IDs
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results.get("IdList", [])
        if not id_list:
            return None
            
        # 2. Fetch Summaries
        summary_handle = Entrez.esummary(db="pubmed", id=",".join(id_list))
        # Use a more flexible parser
        summaries = Entrez.read(summary_handle)
        summary_handle.close()
        
        return summaries
    except Exception as e:
        st.error(f"NCBI Connection Error: {e}")
        return None

# --- UPDATE THE DISPLAY PART IN YOUR MAIN CODE ---
if st.button(f"Fetch Latest Research for '{query}'"):
    with st.spinner("Connecting to NCBI..."):
        data = search_pubmed(query)
        if data:
            # Entrez.read returns a list-like object called 'DocSum'
            for doc in data:
                try:
                    # In esummary, the ID is the key
                    pmid = doc.get('Id', 'Unknown')
                    # Titles are stored in a list of items
                    title = "Untitled Research Paper"
                    pub_date = "Date N/A"
                    
                    for item in doc:
                        if item == 'Title': title = doc[item]
                        if item == 'PubDate': pub_date = doc[item]
                    
                    st.markdown(f"**{title}**")
                    st.write(f"📅 {pub_date} | [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                    st.write("---")
                except:
                    continue 
        else:
            st.warning(f"No results found on PubMed for '{query}'. Try a simpler term.")


# --- UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.markdown("### Specialized Retrieval System for Lehninger Principles of Biochemistry")

# --- SIDEBAR ---
st.sidebar.header("Search & Parameters")
# Added a help tooltip
query = st.sidebar.text_input("Enter Biological Term", placeholder="e.g. Glycolysis, ATP...").lower()

# --- MAIN LOGIC ---
if not query:
    st.info("👈 Enter a term in the sidebar to begin.")
    # Show a preview of the database if loaded
    if df is not None:
        st.write(f"Database loaded successfully with {len(df)} indexed pages.")

elif df is not None:
    # Perform Search
    results = df[df['text_content'].str.contains(query, na=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        
        # Page Selection
        page_list = results['page'].tolist()
        selected_page = st.sidebar.selectbox("Select Page to View", page_list)
        
        # Layout Columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Isolated Diagram")
            diag_url = f"{R2_URL}/diagrams/diag_{selected_page}.png"
            st.image(diag_url, use_container_width=True, caption=f"Figure from Page {selected_page}")
            st.caption("If image is blank, no isolated figure exists for this page.")

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
                        title = "Untitled Article"
                        pub_date = "N/A"
                        for item in art.get('Item', []):
                            if item.attributes.get('Name') == 'Title':
                                title = item
                            if item.attributes.get('Name') == 'PubDate':
                                pub_date = item

                        st.markdown(f"**{title}**")
                        st.write(f"📅 {pub_date} | [Read on PubMed](https://pubmed.ncbi.nlm.nih.gov/{art.get('Id', '')}/)")
                        st.write("---")
                else:
                    st.warning(f"No specific PubMed results for '{query}'.")

        # --- OCR TEXT ---
        with st.expander("View Extracted Page Text (OCR)"):
            page_text = results[results['page'] == selected_page]['text_content'].values[0]
            st.write(page_text)
            
    else:
        st.sidebar.warning(f"No matches found for '{query}'.")
        st.write("Try searching for terms like 'Enzyme', 'Metabolism', or 'Protein'.")
