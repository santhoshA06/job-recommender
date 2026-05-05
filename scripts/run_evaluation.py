from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import POOLED_LABELS_PATH
from src.data_loader import load_clean_dataframes
from src.evaluation import evaluate_run
from src.query_representation import build_query_from_resume
from src.retrievers import TFIDFRetriever, BM25Retriever


def build_run_df(resumes_df: pd.DataFrame, retriever, model_name: str, top_k: int = 10) -> pd.DataFrame:
    rows = []
    for _, row in resumes_df.iterrows():
        resume_id = row["resume_id"]
        query = build_query_from_resume(row["resume_text"])
        results = retriever.search(query, top_k=top_k)
        for rank, (_, result_row) in enumerate(results.iterrows(), start=1):
            rows.append({
                "resume_id": resume_id,
                "job_id": int(result_row["job_id"]),
                "rank": rank,
                "model": model_name,
            })
    return pd.DataFrame(rows)


def main():
    jobs_df, resumes_df = load_clean_dataframes()
    labels_df = pd.read_csv(POOLED_LABELS_PATH)
    labels_df["relevance"] = pd.to_numeric(labels_df["relevance"], errors="coerce").fillna(0).astype(int)

    eval_resume_ids = labels_df["resume_id"].unique().tolist()
    eval_resumes = resumes_df[resumes_df["resume_id"].isin(eval_resume_ids)].copy()

    tfidf = TFIDFRetriever()
    tfidf.load()
    bm25 = BM25Retriever()
    bm25.load()

    tfidf_run = build_run_df(eval_resumes, tfidf, "tfidf", top_k=10)
    bm25_run = build_run_df(eval_resumes, bm25, "bm25", top_k=10)

    tfidf_metrics = evaluate_run(tfidf_run, labels_df, top_k=10)
    bm25_metrics = evaluate_run(bm25_run, labels_df, top_k=10)

    print("TF-IDF Metrics:", tfidf_metrics)
    print("BM25 Metrics:", bm25_metrics)


if __name__ == "__main__":
    main()
