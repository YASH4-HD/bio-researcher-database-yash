import streamlit as st
import pandas as pd
import os

# Set page to wide mode for better side-by-side viewing
st.set_page_config(layout="wide", page_title="BioVisual Search Engine")

# --- LOAD DATA ---
@st.cache_data
def load_index():
    df = pd.read_csv("lehninger_index.csv")
    return df

df = load_index()

# --- UI HEADER ---
st.title("🧬 BioVisual Search Engine")
st.markdown("### Specialized Retrieval System for Lehninger Principles of Biochemistry")

# --- SIDEBAR SEARCH ---
st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Enter Biological Term (e.g., Glycolysis, ATP)", "").lower()

# --- SEARCH LOGIC ---
if query:
    # Filter the dataframe for pages containing the query
    results = df[df['text_content'].str.contains(query, na=False, case=False)]
    
    if not results.empty:
        st.sidebar.success(f"Found in {len(results)} pages")
        
        # Selection box for the specific page found
        page_options = results['page'].tolist()
        selected_page = st.sidebar.selectbox("Select Page to View", page_options)
        
        # --- DISPLAY AREA ---
        col1, col2 = st.columns([1, 1]) # Equal width columns
        
        with col1:
            st.subheader(f"🖼️ Isolated Diagram (Page {selected_page})")
            diag_path = f"lehninger/diagrams/diag_{selected_page}.png"
            if os.path.exists(diag_path):
                st.image(diag_path, use_container_width=True)
            else:
                st.info("No isolated diagram detected for this page.")

        with col2:
            st.subheader(f"📄 Full Page Context")
            full_path = f"lehninger/full_pages/page_{selected_page}.png"
            if os.path.exists(full_path):
                st.image(full_path, use_container_width=True)
            else:
                st.error("Full page image not found.")
                
        # Show text snippet below
        with st.expander("Show Page Text"):
            page_text = results[results['page'] == selected_page]['text_content'].values[0]
            st.write(page_text)
            
    else:
        st.sidebar.warning("No matches found. Try a different term.")
else:
    st.info("👈 Enter a term in the sidebar to begin searching through 4,381 pages.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Researcher: Yashwant Nama")
