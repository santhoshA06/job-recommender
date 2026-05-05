from collections import Counter
from typing import List, Set, Dict, Any
from src.preprocess import preprocess_text, tokenize_text

SKILL_KEYWORDS = [
    "python", "sql", "machine learning", "data science", "nlp",
    "deep learning", "tensorflow", "pytorch", "pandas", "numpy",
    "scikitlearn", "scikit-learn", "tableau", "power bi", "aws",
    "azure", "gcp", "spark", "hadoop", "etl", "data engineering",
    "backend", "api", "flask", "streamlit", "statistics",
    "visualization", "data analysis", "information retrieval", "bm25",
    "tfidf", "embeddings", "database", "mongodb", "mysql", "postgresql"
]

JOB_TITLE_HINTS = [
    "data scientist", "data analyst", "machine learning engineer",
    "data engineer", "business intelligence analyst", "backend developer",
    "software engineer", "research assistant", "information scientist"
]

def extract_skill_matches(text: str) -> List[str]:
    lowered = text.lower()
    found = [skill for skill in SKILL_KEYWORDS if skill in lowered]
    return found

def extract_title_matches(text: str) -> List[str]:
    lowered = text.lower()
    found = [title for title in JOB_TITLE_HINTS if title in lowered]
    return found

def extract_top_terms(text: str, top_n: int = 20) -> List[str]:
    tokens = tokenize_text(text)
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]

def build_query_from_resume(raw_resume_text: str) -> str:
    skills = extract_skill_matches(raw_resume_text)
    titles = extract_title_matches(raw_resume_text)
    top_terms = extract_top_terms(raw_resume_text, top_n=20)

    merged_terms: List[str] = []
    seen: Set[str] = set()

    for item in titles + skills + top_terms:
        normalized = item.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged_terms.append(normalized)

    return preprocess_text(" ".join(merged_terms))

def get_resume_debug_info(raw_resume_text: str) -> Dict[str, Any]:
    return {
        "matched_titles": extract_title_matches(raw_resume_text),
        "matched_skills": extract_skill_matches(raw_resume_text),
        "top_terms": extract_top_terms(raw_resume_text, top_n=15),
        "final_query": build_query_from_resume(raw_resume_text),
    }
    