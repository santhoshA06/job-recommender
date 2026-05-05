from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"

JOBS_RAW_PATH = RAW_DATA_DIR / "linkedin_jobs.csv"
RESUMES_RAW_PATH = RAW_DATA_DIR / "resumes.csv"

JOBS_CLEAN_PATH = PROCESSED_DATA_DIR / "jobs_clean.csv"
RESUMES_CLEAN_PATH = PROCESSED_DATA_DIR / "resumes_clean.csv"
POOLED_LABELS_PATH = PROCESSED_DATA_DIR / "pooled_labels.csv"

TFIDF_VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = ARTIFACT_DIR / "tfidf_matrix.pkl"
BM25_TOKENS_PATH = ARTIFACT_DIR / "bm25_corpus_tokens.pkl"
DENSE_EMBEDDINGS_PATH = ARTIFACT_DIR / "dense_embeddings.npy"
JOBS_METADATA_PATH = ARTIFACT_DIR / "jobs_metadata.csv.gz"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10

for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
