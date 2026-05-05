from typing import Optional, List
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix
from src.config import (
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    BM25_TOKENS_PATH,
    DENSE_EMBEDDINGS_PATH,
    JOBS_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)

METADATA_COLUMNS = ["job_id", "title", "description", "company", "location"]

def _ensure_artifact_ready(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. Run scripts/build_indexes.py first."
        )
    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty at {path}. Rebuild artifacts with scripts/build_indexes.py."
        )

def _save_jobs_metadata(jobs_df: pd.DataFrame) -> None:
    jobs_df.loc[:, METADATA_COLUMNS].to_csv(JOBS_METADATA_PATH, index=False, compression="gzip")

def _load_jobs_metadata() -> pd.DataFrame:
    _ensure_artifact_ready(JOBS_METADATA_PATH, "Jobs metadata")
    return pd.read_csv(JOBS_METADATA_PATH, compression="gzip")

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix: Optional[spmatrix] = None
        self.jobs_df: Optional[pd.DataFrame] = None

    def fit(self, jobs_df: pd.DataFrame):
        self.jobs_df = jobs_df.copy()
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        self.doc_matrix = self.vectorizer.fit_transform(jobs_df["document_text_clean"])

    def save(self):
        assert self.vectorizer is not None
        assert self.doc_matrix is not None
        assert self.jobs_df is not None
        joblib.dump(self.vectorizer, TFIDF_VECTORIZER_PATH)
        joblib.dump(self.doc_matrix, TFIDF_MATRIX_PATH)
        _save_jobs_metadata(self.jobs_df)

    def load(self):
        _ensure_artifact_ready(TFIDF_VECTORIZER_PATH, "TF-IDF vectorizer")
        _ensure_artifact_ready(TFIDF_MATRIX_PATH, "TF-IDF matrix")
        try:
            self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            self.doc_matrix = joblib.load(TFIDF_MATRIX_PATH)
            self.jobs_df = _load_jobs_metadata()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load TF-IDF artifacts. Rebuild them with scripts/build_indexes.py."
            ) from exc

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        assert self.vectorizer is not None
        assert self.doc_matrix is not None
        assert self.jobs_df is not None
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = self.jobs_df.iloc[top_indices].copy()
        results["score"] = scores[top_indices]
        results["model"] = "tfidf"
        return results.reset_index(drop=True)

class BM25Retriever:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: Optional[List[List[str]]] = None
        self.jobs_df: Optional[pd.DataFrame] = None

    def fit(self, jobs_df: pd.DataFrame):
        self.jobs_df = jobs_df.copy()
        self.corpus_tokens = [doc.split() for doc in jobs_df["document_text_clean"].tolist()]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def save(self):
        assert self.corpus_tokens is not None
        assert self.jobs_df is not None
        joblib.dump(self.corpus_tokens, BM25_TOKENS_PATH)
        _save_jobs_metadata(self.jobs_df)

    def load(self):
        _ensure_artifact_ready(BM25_TOKENS_PATH, "BM25 corpus tokens")
        try:
            self.corpus_tokens = joblib.load(BM25_TOKENS_PATH)
            self.jobs_df = _load_jobs_metadata()
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load BM25 artifacts. Rebuild them with scripts/build_indexes.py."
            ) from exc

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        assert self.bm25 is not None
        assert self.jobs_df is not None
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = self.jobs_df.iloc[top_indices].copy()
        results["score"] = scores[top_indices]
        results["model"] = "bm25"
        return results.reset_index(drop=True)

class DenseRetriever:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.jobs_df: Optional[pd.DataFrame] = None

    def fit(self, jobs_df: pd.DataFrame):
        self.jobs_df = jobs_df.copy()
        texts = jobs_df["document_text"].tolist()
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def save(self):
        assert self.embeddings is not None
        assert self.jobs_df is not None
        np.save(DENSE_EMBEDDINGS_PATH, self.embeddings)
        _save_jobs_metadata(self.jobs_df)

    def load(self):
        _ensure_artifact_ready(DENSE_EMBEDDINGS_PATH, "Dense embeddings")
        try:
            self.embeddings = np.load(DENSE_EMBEDDINGS_PATH)
            self.jobs_df = _load_jobs_metadata()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load dense retrieval artifacts. Rebuild them with scripts/build_indexes.py --build_dense."
            ) from exc

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        assert self.embeddings is not None
        assert self.jobs_df is not None
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = self.jobs_df.iloc[top_indices].copy()
        results["score"] = scores[top_indices]
        results["model"] = "dense"
        return results.reset_index(drop=True)
