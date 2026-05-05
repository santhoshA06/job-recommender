import re
import string
from typing import List

import nltk

# Download once
resources = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]

for resource_path, resource_name in resources:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

EXTRA_STOPWORDS = {
    "resume", "curriculum", "vitae", "objective", "summary",
    "experience", "education", "skills", "project", "projects"
}
STOPWORDS = STOPWORDS.union(EXTRA_STOPWORDS)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    return [tok for tok in tokens if tok.isalpha() and tok not in STOPWORDS]


def preprocess_text(text: str) -> str:
    return " ".join(tokenize_text(text))