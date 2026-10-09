---
language: ru
tags:
  - cross-encoder
  - reranking
---
# RuCrossEncoder

Research prototype for Russian query-document relevance and reranking. This
refactor supplies no newly trained weights or reproduced historical scores.
See architecture.md, training.md and evaluation.md.

The report describes DeBERTa; the exact backbone ID/revision, tokenizer changes
and domain-pretraining provenance remain unresolved. Reported classification
scores: accuracy 0.80, precision 0.73, recall 0.94, F1 0.83 (table; prose says 0.82).
These are not verified ranking results or evidence for medical/fraud diagnosis.

The original HF repository https://huggingface.co/ArturAbg/RuCrossEncoder contains
checkpoint.pt. It is not a complete from_pretrained export. New exports include
config.json, safetensors weights and tokenizer files, loaded through this package.
See legacy_checkpoint.md for structural inspection and conditional conversion.

Limitations: synthetic easy negatives, possible factual errors, unresolved
historical thresholds and baseline revisions, no demonstrated domain robustness.
No code license has been selected by the owner. Backbone, data and checkpoint
terms must be checked separately before redistribution.
