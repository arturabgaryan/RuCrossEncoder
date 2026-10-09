"""Explicit scoring contracts for custom models and external baselines."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model import RuCrossEncoderModel


def relevance_scores(logits, positive_label=1, score_mode="softmax"):
    if logits.ndim != 2:
        raise ValueError("Expected [batch, labels] logits")
    if score_mode in {"sigmoid", "raw"}:
        if logits.shape[1] != 1:
            raise ValueError("sigmoid/raw scoring requires exactly one output per pair")
        scores = logits[:, 0]
        return scores.sigmoid() if score_mode == "sigmoid" else scores
    if score_mode != "softmax" or logits.shape[1] < 2:
        raise ValueError("softmax scoring requires at least two output labels")
    if not 0 <= positive_label < logits.shape[1]:
        raise ValueError("positive_label is outside model outputs")
    return logits.softmax(dim=-1)[:, positive_label]


class Reranker:
    def __init__(
        self,
        model_id,
        *,
        kind="custom",
        revision=None,
        tokenizer_id=None,
        tokenizer_revision=None,
        device="auto",
        max_length=128,
        batch_size=32,
        score_mode="softmax",
        positive_label=1,
        use_fast=True,
    ):
        if kind not in {"custom", "sequence_classification"}:
            raise ValueError("kind must be custom or sequence_classification")
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        cls = RuCrossEncoderModel if kind == "custom" else AutoModelForSequenceClassification
        self.model = cls.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id or model_id,
            revision=tokenizer_revision if tokenizer_id else revision,
            use_fast=use_fast,
        )
        self.max_length = max_length
        self.batch_size = batch_size
        self.score_mode = score_mode
        self.positive_label = positive_label

    @torch.inference_mode()
    def score_pairs(self, pairs):
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            queries, texts = zip(*chunk)
            inputs = self.tokenizer(
                list(queries),
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits
            scores.extend(
                relevance_scores(
                    logits,
                    self.positive_label,
                    self.score_mode,
                )
                .cpu()
                .tolist()
            )
        return scores

    def rerank(self, query, documents):
        scores = self.score_pairs([(query, text) for text in documents])
        return sorted(
            [
                {"index": i, "text": text, "score": score}
                for i, (text, score) in enumerate(zip(documents, scores))
            ],
            key=lambda row: row["score"],
            reverse=True,
        )
