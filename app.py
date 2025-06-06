# ESG Document Analyzer Dashboard
# This Streamlit app provides an interface for analyzing ESG documents using RAG and classification
import streamlit as st
from extract_chunk import chunk_pdf_with_spark
from search_pinecone import search_esg
from calculate_scores import compute_esg_score  # or BDA.calculate_scores if inside BDA folder

# Configure the Streamlit page
st.set_page_config("🔍 ESG Insight Dashboard", layout="wide")
st.title("📄 ESG Document Analyzer with RAG + Classification")

# ---- PDF Upload and Processing Section ----
pdf_file = st.file_uploader("Upload a sustainability report PDF", type="pdf")
if pdf_file:
    # Process the uploaded PDF using Spark for efficient chunking
    st.info("🔧 Extracting & chunking PDF using Spark...")
    df_chunks = chunk_pdf_with_spark(pdf_file)
    st.success("PDF successfully chunked.")
    
    # Save and display the chunked data
    df_chunks.toPandas().to_csv("esg_chunk_classification.csv", index=False)
    st.dataframe(df_chunks.toPandas().head())

    # ---- ESG Scoring Section ----
    # Calculate ESG scores based on the extracted chunks
    with st.spinner("🔬 Scoring ESG categories..."):
        final_scores = compute_esg_score("esg_chunk_classification.csv")
        st.success("🎯 ESG Scores calculated!")
        st.dataframe(final_scores)
        
        # Provide download option for the calculated scores
        csv = final_scores.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download ESG Scores", csv, "final_esg_scores.csv", "text/csv")

    # ---- Search Interface Section ----
    st.markdown("---")
    st.header("🔎 Ask your ESG question")
    
    # Search input components
    query = st.text_input("Enter your query", placeholder="e.g., What are the Scope 3 emissions?")
    category = st.selectbox("Filter by ESG category (optional)", 
                          ["", "Scope 1", "Scope 2", "Scope 3", 
                           "Water Reduction", "Sustainable Packaging", "Governance"])
    k = st.slider("Top-k results", 1, 10, 3)  # Number of results to display

    # Execute search when button is clicked
    if st.button("Search"):
        st.info("🧠 Querying Pinecone for matches...")
        results = search_esg(query, top_k=k, category=category if category else None)

        # Display search results with metadata
        for i, match in enumerate(results):
            st.markdown(f"### 🔹 Match {i+1} — Score: `{match['score']:.2f}`")
            st.write(match["metadata"]["text"])
            if "category" in match["metadata"]:
                st.markdown(f"**Category:** {match['metadata']['category']} &nbsp;&nbsp; | &nbsp;&nbsp; **Confidence:** `{match['metadata'].get('confidence', 'N/A')}`")
            st.markdown("---")
