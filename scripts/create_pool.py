from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import POOLED_LABELS_PATH
from src.data_loader import load_clean_dataframes
from src.query_representation import build_query_from_resume
from src.retrievers import TFIDFRetriever, BM25Retriever


def main():
    jobs_df, resumes_df = load_clean_dataframes()

    # Use only a subset for evaluation
    sample_resumes = resumes_df.head(20).copy()

    tfidf = TFIDFRetriever()
    tfidf.load()

    bm25 = BM25Retriever()
    bm25.load()

    pooled_rows = []

    for _, row in sample_resumes.iterrows():
        resume_id = row["resume_id"]
        query = build_query_from_resume(row["resume_text"])

        tfidf_results = tfidf.search(query, top_k=10)
        bm25_results = bm25.search(query, top_k=10)

        pooled = pd.concat([tfidf_results, bm25_results], ignore_index=True)
        pooled = pooled.drop_duplicates(subset=["job_id"]).reset_index(drop=True)

        for rank, (_, result_row) in enumerate(pooled.iterrows(), start=1):
            pooled_rows.append({
                "resume_id": resume_id,
                "job_id": int(result_row["job_id"]),
                "title": result_row["title"],
                "company": result_row.get("company", ""),
                "relevance": "",   # Fill manually: 0, 1, or 2
                "notes": "",
                "rank": rank,
            })

    pool_df = pd.DataFrame(pooled_rows)
    pool_df.to_csv(POOLED_LABELS_PATH, index=False)
    print(f"Saved pool file to {POOLED_LABELS_PATH}")
    print("Open it in Excel and manually label relevance: 0=not relevant, 1=partially relevant, 2=relevant")


if __name__ == "__main__":
    main()
