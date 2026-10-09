"""Classification and ranking are reported separately."""

import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from .data import read_records, validate_triples
from .inference import Reranker
from .utils import manifest, write_json


def classification_metrics(labels, scores, threshold):
    if not len(labels) or len(labels) != len(scores):
        raise ValueError("Labels and scores must have the same nonzero length")
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    preds = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    p, r, thresholds = precision_recall_curve(labels, scores)
    return {
        "pairs": len(labels),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "pr_curve": {
            "precision": p.tolist(),
            "recall": r.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def ranking_metrics(relevance, scores, k=10):
    relevance = np.asarray(relevance, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if k < 1 or len(scores) == 0 or len(scores) != len(relevance):
        raise ValueError("Ranking requires nonempty aligned scores/relevance and positive k")
    if not np.isfinite(scores).all() or not np.isfinite(relevance).all() or (relevance < 0).any():
        raise ValueError("Invalid scores or relevance grades")
    ranked = relevance[np.argsort(-scores, kind="stable")][:k]
    ideal = np.sort(relevance)[::-1][:k]
    discount = np.log2(np.arange(len(ranked)) + 2)
    dcg = np.sum((2**ranked - 1) / discount)
    idcg = np.sum((2**ideal - 1) / discount)
    hits = ranked > 0
    total_positive = int((relevance > 0).sum())
    hit_positions = np.flatnonzero(hits)
    ap = np.sum((np.cumsum(hits) / (np.arange(len(hits)) + 1)) * hits)
    return {
        f"ndcg@{k}": float(dcg / idcg) if idcg else 0.0,
        f"mrr@{k}": float(1 / (hit_positions[0] + 1)) if len(hit_positions) else 0.0,
        f"map@{k}": float(ap / min(total_positive, k)) if total_positive else 0.0,
        f"recall@{k}": float(hits.sum() / total_positive) if total_positive else 0.0,
    }


def evaluate(config):
    rows = read_records(config["test_path"])
    mode = config.get("mode", "classification")
    if mode == "classification":
        validate_triples(rows)
        pairs = [(row["query"], row[field]) for row in rows for field in ("positive", "negative")]
        labels = [label for _ in rows for label in (1, 0)]
    elif mode == "ranking":
        for row in rows:
            if (
                not isinstance(row.get("query"), str)
                or not row["query"].strip()
                or not row.get("docs")
            ):
                raise ValueError("Ranking records require a query and nonempty docs list")
            for doc in row["docs"]:
                if not isinstance(doc.get("text"), str) or not doc["text"].strip():
                    raise ValueError("Each document needs nonempty text")
                if not isinstance(doc.get("relevance"), (int, float)) or doc["relevance"] < 0:
                    raise ValueError("Each document needs nonnegative numeric relevance")
    else:
        raise ValueError("mode must be classification or ranking")
    if not config.get("models"):
        raise ValueError("Provide at least one model in benchmark configuration")
    results = {}
    for spec in config["models"]:
        if spec["name"] in results:
            raise ValueError("Model names must be unique")
        threshold = spec.get("threshold", 0.5)
        model_options = {k: v for k, v in spec.items() if k not in {"name", "threshold"}}
        reranker = Reranker(**model_options)
        if mode == "classification":
            results[spec["name"]] = classification_metrics(
                labels, reranker.score_pairs(pairs), threshold
            )
        else:
            metrics = []
            for row in rows:
                scores = reranker.score_pairs([(row["query"], doc["text"]) for doc in row["docs"]])
                metrics.append(
                    ranking_metrics(
                        [doc["relevance"] for doc in row["docs"]], scores, config.get("k", 10)
                    )
                )
            results[spec["name"]] = {
                key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]
            }
            results[spec["name"]]["queries"] = len(rows)
        print(
            spec["name"],
            {k: v for k, v in results[spec["name"]].items() if k != "pr_curve"},
            flush=True,
        )
        del reranker
    output = Path(config.get("output_dir", "artifacts/evaluation"))
    write_json(results, output / "metrics.json")
    run_manifest = manifest(config)
    run_manifest["test_sha256"] = hashlib.sha256(Path(config["test_path"]).read_bytes()).hexdigest()
    write_json(run_manifest, output / "manifest.json")
    return results
