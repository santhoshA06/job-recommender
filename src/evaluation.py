from typing import Dict, List

import numpy as np
import pandas as pd


def precision_at_k(relevances: List[int], k: int = 10) -> float:
    rel_k = relevances[:k]
    binary = [1 if r > 0 else 0 for r in rel_k]
    return float(np.mean(binary)) if binary else 0.0


def reciprocal_rank(relevances: List[int]) -> float:
    for i, r in enumerate(relevances, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: List[int], k: int = 10) -> float:
    rel_k = relevances[:k]
    return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rel_k))


def ndcg_at_k(relevances: List[int], k: int = 10) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_run(run_df: pd.DataFrame, labels_df: pd.DataFrame, top_k: int = 10) -> Dict[str, float]:
    # Keep only the columns needed from labels to avoid rank collisions
    labels_subset = labels_df[["resume_id", "job_id", "relevance"]].copy()

    merged = run_df.merge(labels_subset, on=["resume_id", "job_id"], how="left")
    merged["relevance"] = merged["relevance"].fillna(0).astype(int)

    metrics = []
    for resume_id, group in merged.groupby("resume_id"):
        group = group.sort_values("rank")
        rels = group["relevance"].tolist()

        metrics.append({
            "resume_id": resume_id,
            "p_at_k": precision_at_k(rels, top_k),
            "rr": reciprocal_rank(rels),
            "ndcg_at_k": ndcg_at_k(rels, top_k),
        })

    metrics_df = pd.DataFrame(metrics)

    return {
        "mean_precision_at_k": float(metrics_df["p_at_k"].mean()) if not metrics_df.empty else 0.0,
        "mrr": float(metrics_df["rr"].mean()) if not metrics_df.empty else 0.0,
        "mean_ndcg_at_k": float(metrics_df["ndcg_at_k"].mean()) if not metrics_df.empty else 0.0,
    }