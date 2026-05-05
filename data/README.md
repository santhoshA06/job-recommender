# Data Description

## Overview

This project uses two datasets:

1. A job postings dataset used as the document collection
2. A resume dataset used as the query collection

The datasets are processed and indexed to support TF-IDF and BM25 retrieval models for resume-based job recommendation.

---

## Dataset 1: LinkedIn Job Postings Dataset

The job postings dataset contains job advertisements collected from online recruitment platforms. Each record includes information such as job title, company, location, skills and detailed job descriptions.

### Dataset Source

LinkedIn Job Postings (2023–2024)

https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

### Expected File Path

```text
data/raw/linkedin_jobs.csv
```

### Main Fields

- Job Title
- Company Name
- Location
- Job Description
- Skills
- Employment Information

### Purpose

This dataset serves as the document corpus for the Information Retrieval system. Each job posting is treated as a searchable document for retrieval and ranking.

---

## Dataset 2: Resume Dataset

The resume dataset contains resumes from different candidates with information related to education, technical skills, certifications, and work experience.

### Dataset Source

Resume Dataset

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

### Expected File Path

```text
data/raw/resumes.csv
```

### Main Fields

- Resume Text
- Candidate Skills
- Education
- Work Experience
- Resume Category

### Purpose

This dataset is used as the query collection. Each resume is treated as a long query for retrieving relevant job postings from the job corpus.

---

## Why Raw Data Is Not Included

The original datasets are not included in this repository because of GitHub file size limitations and storage constraints.

Users should download the datasets separately from the provided sources and place them in the required directories.

---

## Data Preprocessing

The preprocessing pipeline performs several text normalization operations, including text cleaning, lowercasing, tokenization, stopword removal, and resume normalization. These steps improve query representation and retrieval effectiveness.

To preprocess the datasets, run:

```bash
python -m scripts.prepare_data
```

---

## Processed Data

The preprocessing pipeline generates cleaned datasets inside:

```text
data/processed/
```

Generated files include:

```text
jobs_clean.csv
resumes_clean.csv
pooled_labels.csv
```

These files are generated locally and excluded from Git tracking using `.gitignore`.

---

## Retrieval Indexes

The project supports two retrieval models:

- TF-IDF
- BM25

Retrieval indexes can be generated using:

```bash
python -m scripts.build_indexes
```

---

## Evaluation Data

The project uses pooling-based relevance judgments for evaluation because the resume and job datasets are independent collections without predefined relevance labels.

To create pooled retrieval results:

```bash
python -m scripts.create_pool
```

To evaluate retrieval performance:

```bash
python -m scripts.run_evaluation
```

Evaluation metrics include:

- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG@K

---

## Ignored Data Folders

The following folders are excluded from GitHub using `.gitignore`:

```text
data/raw/
data/processed/
data/artifacts/
```

These files can be regenerated locally using the provided preprocessing and indexing scripts.