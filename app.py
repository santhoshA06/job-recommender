import pandas as pd
import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.query_representation import build_query_from_resume, get_resume_debug_info
from src.retrievers import TFIDFRetriever, BM25Retriever
from src.utils import snippet

st.set_page_config(page_title="Resume to Job Retrieval", layout="wide")

st.title("Resume-Based Job Recommendation System")
st.write(
    "Upload a text-based PDF resume. The system extracts text, builds a focused query "
    "representation, and retrieves relevant job postings using information retrieval models."
)

@st.cache_resource
def load_retrievers():
    tfidf = TFIDFRetriever()
    tfidf.load()

    bm25 = BM25Retriever()
    bm25.load()

    return tfidf, bm25


try:
    tfidf, bm25 = load_retrievers()
except Exception as exc:
    st.error(
        "Retrieval indexes are not ready yet. Run `python scripts/prepare_data.py` "
        "and then `python scripts/build_indexes.py`, then refresh the app."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

model_choice = st.selectbox(
    "Choose retrieval model",
    ["tfidf", "bm25"],
)

uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text.strip():
        st.error("No text could be extracted. Please upload a text-based PDF.")
        st.stop()

    st.subheader("Extracted Resume Preview")
    st.text_area("Resume Text", resume_text[:3000], height=250)

    debug_info = get_resume_debug_info(resume_text)
    final_query = debug_info["final_query"]

    st.subheader("Generated Query Representation")
    st.write(final_query)

    with st.expander("Debug: extracted titles, skills, and top terms"):
        st.write("Matched Titles:", debug_info["matched_titles"])
        st.write("Matched Skills:", debug_info["matched_skills"])
        st.write("Top Terms:", debug_info["top_terms"])

    with st.spinner("Retrieving jobs..."):
        if model_choice == "tfidf":
            results = tfidf.search(final_query, top_k=10)
        else:
            results = bm25.search(final_query, top_k=10)

    st.subheader("Top Recommended Job Postings")

    if results.empty:
        st.warning("No matching jobs found.")
    else:
        display_rows = []
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            display_rows.append({
                "Rank": rank,
                "Job Title": row["title"],
                "Company": row.get("company", ""),
                "Location": row.get("location", ""),
                "Score": round(float(row["score"]), 4),
                "Snippet": snippet(row["description"], 200),
            })

        st.dataframe(pd.DataFrame(display_rows), use_container_width=True)

        st.subheader("Detailed Results")
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {row['title']}**")
                if row.get("company", ""):
                    st.write(f"Company: {row['company']}")
                if row.get("location", ""):
                    st.write(f"Location: {row['location']}")
                st.write(f"Score: {round(float(row['score']), 4)}")
                st.write(snippet(row["description"], 700))
