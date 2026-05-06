import os
import math
import logging
from io import BytesIO
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
from src.pdf_utils import extract_text_from_pdf
from src.query_representation import build_query_from_resume, get_resume_debug_info
from src.retrievers import TFIDFRetriever, BM25Retriever
from src.utils import snippet

app = FastAPI(title="Resume Job Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tfidf = TFIDFRetriever()
bm25 = BM25Retriever()
startup_complete = False

@app.on_event("startup")
def startup_event():
    global startup_complete
    tfidf.load()
    bm25.load()
    startup_complete = True

def filter_results(df: pd.DataFrame, location_filter: str = "", title_filter: str = "") -> pd.DataFrame:
    filtered = df.copy()

    if location_filter and location_filter.strip():
        filtered = filtered[
            filtered["location"]
            .fillna("")
            .astype(str)
            .str.contains(location_filter.strip(), case=False, na=False)
        ]

    if title_filter and title_filter.strip():
        filtered = filtered[
            filtered["title"]
            .fillna("")
            .astype(str)
            .str.contains(title_filter.strip(), case=False, na=False)
        ]

    return filtered.reset_index(drop=True)

@app.get("/health")
def health_check():
    return {"status": "ok", "startup_complete": startup_complete}

@app.post("/recommend")
async def recommend_jobs(
    file: UploadFile = File(...),
    model: str = Form("tfidf"),
    top_k: int = Form(10),
    location_filter: str = Form(""),
    title_filter: str = Form(""),
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF files are supported."},
            )

        file_bytes = await file.read()
        pdf_file = BytesIO(file_bytes)

        resume_text = extract_text_from_pdf(pdf_file)

        if not resume_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No text could be extracted from the PDF. Please upload a text-based PDF."},
            )

        debug_info = get_resume_debug_info(resume_text)
        final_query = build_query_from_resume(resume_text)

        # Retrieve more results first, then filter, then keep top_k
        search_k = max(top_k * 5, 50)

        if model == "tfidf":
            results = tfidf.search(final_query, top_k=search_k)
        elif model == "bm25":
            results = bm25.search(final_query, top_k=search_k)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid model. Use 'tfidf' or 'bm25'."},
            )

        print("Before filtering:", len(results))
        print("Location filter:", repr(location_filter))
        print("Title filter:", repr(title_filter))

        results = filter_results(results, location_filter, title_filter)
        results = results.head(top_k).reset_index(drop=True)

        print("After filtering:", len(results))

        response_rows = []
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            raw_score = row.get("score", 0.0)

            try:
                safe_score = float(raw_score)
                if math.isnan(safe_score) or math.isinf(safe_score):
                    safe_score = 0.0
            except (TypeError, ValueError):
                safe_score = 0.0

            response_rows.append(
                {
                    "rank": rank,
                    "title": str(row.get("title", "")),
                    "company": str(row.get("company", "")),
                    "location": str(row.get("location", "")),
                    "score": safe_score,
                    "snippet": snippet(str(row.get("description", "")), 250),
                    "description": snippet(str(row.get("description", "")), 1000),
                }
            )

        return {
            "model": model,
            "query": final_query,
            "matched_titles": debug_info.get("matched_titles", []),
            "matched_skills": debug_info.get("matched_skills", []),
            "top_terms": debug_info.get("top_terms", []),
            "resume_preview": resume_text[:4000],
            "results": response_rows,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )