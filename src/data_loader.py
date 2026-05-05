from pathlib import Path
from typing import Tuple
import pandas as pd
from src.config import JOBS_RAW_PATH, RESUMES_RAW_PATH, JOBS_CLEAN_PATH, RESUMES_CLEAN_PATH
from src.preprocess import preprocess_text

def _read_required_csv(path: Path, label: str, hint: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}. {hint}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty at {path}. {hint}")
    return pd.read_csv(path)

def load_raw_jobs() -> pd.DataFrame:
    return _read_required_csv(JOBS_RAW_PATH, "Raw jobs data", "Check data/raw/linkedin_jobs.csv.")

def load_raw_resumes() -> pd.DataFrame:
    return _read_required_csv(RESUMES_RAW_PATH, "Raw resumes data", "Check data/raw/resumes.csv.")

def clean_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Adjust these column names if your Kaggle file differs
    title_col = None
    desc_col = None
    for col in df.columns:
        low = col.lower()
        if title_col is None and "title" in low:
            title_col = col
        if desc_col is None and ("description" in low or "desc" in low):
            desc_col = col
    if title_col is None or desc_col is None:
        raise ValueError(
            f"Could not find title/description columns. Columns found: {list(df.columns)}"
        )

    cleaned = pd.DataFrame()
    cleaned["job_id"] = range(len(df))
    cleaned["title"] = df[title_col].fillna("").astype(str)
    cleaned["description"] = df[desc_col].fillna("").astype(str)

    company_col = next((c for c in df.columns if "company" in c.lower()), None)
    location_col = next((c for c in df.columns if "location" in c.lower()), None)

    cleaned["company"] = df[company_col].fillna("").astype(str) if company_col else ""
    cleaned["location"] = df[location_col].fillna("").astype(str) if location_col else ""

    cleaned["document_text"] = (
        cleaned["title"] + " " + cleaned["description"]
    ).str.strip()

    cleaned["document_text_clean"] = cleaned["document_text"].apply(preprocess_text)
    cleaned = cleaned[cleaned["document_text_clean"].str.len() > 0].reset_index(drop=True)
    return cleaned

def clean_resumes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Flexible matching for resume/category columns
    resume_col = None
    category_col = None

    for col in df.columns:
        low = col.lower()
        if resume_col is None and ("resume" in low or "text" in low):
            resume_col = col
        if category_col is None and ("category" in low or "label" in low):
            category_col = col

    if resume_col is None:
        raise ValueError(f"Could not find resume text column. Columns found: {list(df.columns)}")

    cleaned = pd.DataFrame()
    cleaned["resume_id"] = range(len(df))
    cleaned["category"] = df[category_col].fillna("").astype(str) if category_col else ""
    cleaned["resume_text"] = df[resume_col].fillna("").astype(str)
    cleaned["resume_text_clean"] = cleaned["resume_text"].apply(preprocess_text)
    cleaned = cleaned[cleaned["resume_text_clean"].str.len() > 0].reset_index(drop=True)
    return cleaned

def save_clean_dataframes(jobs_df: pd.DataFrame, resumes_df: pd.DataFrame) -> None:
    jobs_df.to_csv(JOBS_CLEAN_PATH, index=False)
    resumes_df.to_csv(RESUMES_CLEAN_PATH, index=False)

def load_clean_jobs() -> pd.DataFrame:
    return _read_required_csv(
        JOBS_CLEAN_PATH,
        "Clean jobs data",
        "Run scripts/prepare_data.py first.",
    )

def load_clean_resumes() -> pd.DataFrame:
    return _read_required_csv(
        RESUMES_CLEAN_PATH,
        "Clean resumes data",
        "Run scripts/prepare_data.py first.",
    )

def load_clean_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    jobs_df = load_clean_jobs()
    resumes_df = load_clean_resumes()
    return jobs_df, resumes_df
