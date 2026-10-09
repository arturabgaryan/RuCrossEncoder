# Evaluation protocol

Historical scores in results/report_2025/metrics.json are reported, not reproduced.
The original code does not establish exact models, tokenizers, thresholds or
dataset splits used for that table.

Pin data, model revisions and config, then run
`rucrossencoder evaluate --config configs/benchmark.yaml`.
Each baseline gets its own tokenizer and scoring contract: softmax plus
positive_label for multiclass outputs, sigmoid for one binary logit, or raw for
an untransformed single score. The adapter handles HF sequence classifiers,
not embedding models. Select thresholds on dev data and freeze before testing.

Classification reports pair count, accuracy, precision, recall, F1, average
precision, confusion matrix and PR curves. High recall at lower precision does
not establish superiority. Compare PR curves or recall at a fixed precision.

Ranking reports query-mean nDCG@k, MRR@k, MAP@k and Recall@k. Grades >0 count as
relevant for binary metrics; nDCG uses gain 2**grade - 1. MAP@k divides by
min(total relevant, k). Ties preserve input order; queries without positives
contribute zero. Two easy candidates are not representative of retrieval;
use fixed pools from a real first-stage search system.

New outputs include metrics and a manifest with test SHA256. Record dataset
provenance, candidate generation, hardware and runtime before publishing.
Domain-specific claims require evaluations in those domains.
