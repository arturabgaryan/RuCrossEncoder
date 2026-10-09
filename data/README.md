# Data contracts

Training/classification: JSONL records with nonempty `query`, `positive`,
`negative` strings. Each triple becomes two labeled pairs. CSV, XLSX and Parquet
with these columns are also accepted (install `.[data]` for XLSX/Parquet).
Ranking: JSONL records with `query` and a nonempty `docs` list; each document has
`text` and a nonnegative `relevance` grade. This is a different task.

`samples/` contains hand-written examples, not a benchmark. `raw/` and
`processed/` are local and ignored by Git.

Historical sources:
- https://huggingface.co/datasets/unicamp-dl/mmarco
- https://huggingface.co/datasets/ArturAbg/GeneratedRURerank

Download/export the actual data into a supported local format. The original
`load_dataset()` result must not be passed as a pandas DataFrame. This package
does not execute legacy Hugging Face dataset scripts automatically.
Record source revision, license, download command and SHA256 per experiment.
The split command groups case/whitespace query variants, not paraphrases.
Generated answers need manual review: JSON validity does not imply factual accuracy.
