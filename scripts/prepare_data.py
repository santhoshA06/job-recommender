from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_raw_jobs,
    load_raw_resumes,
    clean_jobs_dataframe,
    clean_resumes_dataframe,
    save_clean_dataframes,
)


def main():
    jobs_raw = load_raw_jobs()
    resumes_raw = load_raw_resumes()

    jobs_clean = clean_jobs_dataframe(jobs_raw)
    resumes_clean = clean_resumes_dataframe(resumes_raw)

    save_clean_dataframes(jobs_clean, resumes_clean)

    print("Jobs cleaned:", jobs_clean.shape)
    print("Resumes cleaned:", resumes_clean.shape)
    print("Saved cleaned datasets.")


if __name__ == "__main__":
    main()
