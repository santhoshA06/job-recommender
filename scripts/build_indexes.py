import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_clean_jobs
from src.retrievers import TFIDFRetriever, BM25Retriever, DenseRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dense", action="store_true", help="Build dense embeddings too")
    args = parser.parse_args()

    jobs_df = load_clean_jobs()

    print("Building TF-IDF...")
    tfidf = TFIDFRetriever()
    tfidf.fit(jobs_df)
    tfidf.save()

    print("Building BM25...")
    bm25 = BM25Retriever()
    bm25.fit(jobs_df)
    bm25.save()

    if args.build_dense:
        print("Building dense embeddings...")
        dense = DenseRetriever()
        dense.fit(jobs_df)
        dense.save()

    print("Done.")


if __name__ == "__main__":
    main()
