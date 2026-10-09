# Training

1. Install the package; obtain the actual backbone, tokenizer and source data.
2. Export records into the documented local format.
3. Run `rucrossencoder split --input data/raw/triples.jsonl --output-dir data/processed --seed 42`.
4. Set encoder_id and revisions in configs/train.yaml.
5. Run `rucrossencoder train --config configs/train.yaml`.

Training checks query disjointness and uses dynamic padding, AdamW with explicit
weight decay, linear warmup/decay, clipping and real gradient accumulation.
FP16 is enabled on CUDA only. Dev loss selects best/; last/ stores the last export.
Both exports include the tokenizer. Large artifacts are ignored by Git.

resume.pt contains optimizer, scheduler, scaler and RNG states. Set resume_from
to continue the same run at the next epoch. Data hashes and training settings
must match. Only load trusted locally created resume files; RNG restoration
requires Python deserialization.

Manifests record config, commit, dirty state, package/platform/CUDA versions and
data hashes. Pin model revisions. Seeds do not guarantee bitwise equality across
devices or library versions. Tests use a tiny local BERT, not historical weights.

## Generation

Install `.[generate]`, start Ollama and fetch the model, then run
`rucrossencoder generate --config configs/generate.yaml`.
Generation writes incremental JSONL, resumes from existing records and retries
malformed outputs/duplicate queries. Cyrillic checks do not guarantee Russian-only
text or correct answers. Easy unrelated negatives preserve the historical prompt;
hard-negative mining remains separate work.
