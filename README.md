# RuCrossEncoder

Russian query-document relevance classification and reranking with a joint
Transformer encoder and an MLP classification head.

**Status:** research prototype being made reproducible. The original report
describes DeBERTa; its exact backbone/tokenizer remain unresolved. This refactor
does not claim new training results or reproduce the historical benchmark.

## Installation

Python 3.10+ is required. Install PyTorch for your hardware first if needed, then:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e '.[dev]'
```

Optional extras: `.[generate]` for Ollama; `.[data]` for XLSX/Parquet.
With uv, `uv sync --locked --extra dev` installs the checked-in dependency lock.
The lock uses the default PyPI PyTorch distribution; select a hardware-specific
PyTorch source in your deployment environment as needed. CPU pipeline tests are
also compatible with PyTorch's CPU-only distribution.

## Use a complete model export

```python
from rucrossencoder.inference import Reranker

ranker = Reranker("artifacts/training/best")
print(
    ranker.rerank(
        "Как поменять пароль?",
        ["Откройте настройки аккаунта.", "Завтра ожидается дождь."],
    )
)
```

This requires trained/exported weights, config and tokenizer in one directory.
The [legacy HF checkpoint](https://huggingface.co/ArturAbg/RuCrossEncoder) is not
yet a complete loadable export. See [model card](docs/model_card.md).
The checkpoint's known dimensions and missing tokenizer are documented in
[legacy recovery notes](docs/legacy_checkpoint.md). An export command is available
when a compatible original configuration and tokenizer have been recovered.

## Train and evaluate

```bash
rucrossencoder split --input data/raw/triples.jsonl --output-dir data/processed --seed 42
# Set the actual encoder_id and revisions in configs/train.yaml first.
rucrossencoder train --config configs/train.yaml
rucrossencoder evaluate --config configs/benchmark.yaml
rucrossencoder rerank --model artifacts/training/best --input examples/rerank_input.json
```

Triples have query, positive, negative columns. Splits group normalized queries.
Dev loss selects a checkpoint; classification and ranking are separate tasks.
See [data contracts](data/README.md), [training](docs/training.md),
[evaluation](docs/evaluation.md), [architecture](docs/architecture.md).

## Historical reported results

From the author's internship report, section 2.7, PDF page 22.
**Reported, not reproduced.** Exact baseline revisions and thresholds remain
unresolved. The report describes 200,000 Russian text pairs.

| Report model label | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| MsMarco | 0.91 | 0.97 | 0.85 | 0.91 |
| FRIDA | 0.88 | 0.86 | 0.90 | 0.88 |
| RuCrossEncoder / MyModel | 0.80 | 0.73 | 0.94 | 0.83 |

F1 is 0.83 in the table and 0.82 in adjacent prose. Higher recall at lower
precision does not establish overall or ranking superiority. Machine-readable
values: [metrics.json](results/report_2025/metrics.json).

Data: [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) and
[generated Russian triples](https://huggingface.co/datasets/ArturAbg/GeneratedRURerank).
The generation prompt uses unrelated negatives. 

## Repository

- src/rucrossencoder/: model, data, training, inference, evaluation, generation, CLI.
- configs/: explicit experiment parameters.
- tests/: local CPU tests, including tiny-model training and serialization.
- examples/ and data/samples/: illustrative inputs.
- docs/ and results/: protocol, model card and historical scores.
- notebooks/legacy/: original notebooks retained for reference.

Large data, weights and run artifacts are ignored by Git. Do not commit credentials.
