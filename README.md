# Resume-Based Job Recommendation System Using IR

## Overview

This project is an Information Retrieval (IR) system that recommends relevant job postings based on an uploaded resume. The system treats a resume as a long query, extracts meaningful text signals, and retrieves the most relevant jobs using traditional IR retrieval models.

The project supports two retrieval methods:

- TF-IDF
- BM25

A React frontend is used for the user interface, while a FastAPI backend handles PDF processing, query generation, retrieval, and ranking. A Streamlit-based interface is also included for quick experimentation and testing.

---

## Features

- Upload a resume PDF
- Extract resume text automatically
- Generate focused query representations
- Retrieve relevant job postings
- Compare TF-IDF and BM25 retrieval models
- Display ranked recommendations with similarity scores
- Support both React UI and Streamlit UI
- Evaluate retrieval performance using IR metrics

---

## Project Structure

```text
job-recommender/
│
├── backend/
│   ├── main.py
│   ├── __init__.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── pdf_utils.py
│   ├── preprocess.py
│   ├── query_representation.py
│   ├── retrievers.py
│   └── utils.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_indexes.py
│   ├── create_pool.py
│   └── run_evaluation.py
│
├── data/
│   └── README.md
│
├── docs/
│   ├── api.md
│   └── execution_steps.md
│
├── requirements.txt
├── app.py
├── .gitignore
└── README.md
```

---

## System Architecture

```text
React Frontend / Streamlit UI
                ↓
          FastAPI Backend
                ↓
      Resume PDF Text Extraction
                ↓
        Query Representation
                ↓
      TF-IDF / BM25 Retrieval
                ↓
     Ranked Job Recommendations
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/santhoshA06/job-recommender.git
cd job-recommender
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Data Setup

The raw datasets are not included in this repository because of file size limitations.

Expected dataset locations:

```text
data/raw/linkedin_jobs.csv
data/raw/resumes.csv
```

Dataset details are provided in:

```text
data/README.md
```

After placing the datasets in the correct paths, run:

```bash
python -m scripts.prepare_data
```

Then build retrieval indexes:

```bash
python -m scripts.build_indexes
```

---

## Running the Application

This project supports two interfaces:

1. React + FastAPI full-stack UI
2. Streamlit UI

---

### Option 1: React + FastAPI Interface

Start the backend from the project root:

```bash
uvicorn backend.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Open a second terminal and start the frontend:

```bash
cd frontend
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

### Option 2: Streamlit Interface

Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

Streamlit URL:

```text
http://localhost:8501
```

---

## Usage

1. Upload a resume PDF
2. Select TF-IDF or BM25 retrieval
3. Choose the number of top results
4. Click **Get Recommendations**
5. View ranked job recommendations and similarity scores

---

## Evaluation

The system uses pooled relevance judgments for evaluation because the resume and job datasets are independent collections.

Evaluation metrics include:

- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG@K

To create pooled labels:

```bash
python -m scripts.create_pool
```

To run evaluation:

```bash
python -m scripts.run_evaluation
```

---

## Key IR Concepts Used

This project applies several Information Retrieval concepts:

- Resume as a long query
- Query preprocessing and normalization
- Vector Space Model using TF-IDF
- Probabilistic retrieval using BM25
- Pooling-based relevance judgments
- Ranking evaluation metrics

---

## Technologies Used

### Backend
- Python
- FastAPI
- Pandas
- Scikit-learn
- NLTK

### Frontend
- React
- Vite
- CSS

### Additional Tools
- Streamlit
- GitHub
- VS Code

---

## Author

Santhosh Adavala

University of Arizona  
MS in Data Science