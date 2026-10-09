# RuCrossEncoder

I developed RuCrossEncoder to evaluate the relevance of Russian query-document pairs and rerank search results. My model combines a DeBERTa encoder with a custom classification head and processes both texts together to capture their semantic relationship.

## Model

I use the encoder's pooled representation, or the first token embedding when a pooled output is unavailable. My classification head consists of a linear layer, GELU activation, layer normalization, dropout (0.1), and a final linear layer that produces relevance logits.

For binary relevance classification, I train the model with cross-entropy loss. My training pipeline uses AdamW, linear learning-rate warmup and decay, gradient clipping, optional mixed precision, and checkpoints for resuming training.

## Data and results

I evaluated my model on a benchmark of 200,000 Russian text pairs and compared it with MsMarco and FRIDA.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| MsMarco | 0.91 | 0.97 | 0.85 | 0.91 |
| FRIDA | 0.88 | 0.86 | 0.90 | 0.88 |
| RuCrossEncoder | 0.80 | 0.73 | 0.94 | 0.83 |

My model achieved the highest recall among the three models in this comparison. This reflects a trade-off: it identifies more relevant pairs, with lower precision and accuracy than MsMarco.

For Russian benchmark data, I built a generation pipeline using DeepSeek-R1 and validated records with Pydantic. Each record contains a query, a relevant answer, and an unrelated negative example. I removed duplicates and invalid records before splitting the data.

My model and datasets are available on Hugging Face:

- [RuCrossEncoder checkpoint](https://huggingface.co/ArturAbg/RuCrossEncoder)
- [GeneratedRURerank dataset](https://huggingface.co/datasets/ArturAbg/GeneratedRURerank)
- [mMARCO dataset](https://huggingface.co/datasets/unicamp-dl/mmarco)

## Installation

I use Python 3.10 or later. Install the appropriate PyTorch distribution for your hardware, then install the package:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e '.[dev]'
```

Optional extras: `.[generate]` for Ollama and `.[data]` for XLSX/Parquet support.

With uv:

```bash
uv sync --locked --extra dev
```

## Training and evaluation

My package includes commands for splitting data, training, evaluation, and reranking:

```bash
rucrossencoder split --input data/raw/triples.jsonl --output-dir data/processed --seed 42
rucrossencoder train --config configs/train.yaml
rucrossencoder evaluate --config configs/benchmark.yaml
rucrossencoder rerank --model artifacts/training/best --input examples/rerank_input.json
```

Before training, set `encoder_id`, the tokenizer settings, model revisions, and data paths in `configs/train.yaml`. Input triples use the `query`, `positive`, and `negative` fields. The split command groups normalized queries, and training selects the best checkpoint by development loss.

I document the input formats in [data contracts](data/README.md), the training setup in [training](docs/training.md), and the evaluation commands in [evaluation](docs/evaluation.md).

## Reranking

I use a model directory containing the saved weights, configuration, and tokenizer:

```python
from rucrossencoder.inference import Reranker

ranker = Reranker("artifacts/training/best")
results = ranker.rerank(
    "Как поменять пароль?",
    ["Откройте настройки аккаунта.", "Завтра ожидается дождь."],
)
print(results)
```

## Repository structure

- `src/rucrossencoder/`: model, data processing, training, inference, evaluation, generation, and CLI.
- `configs/`: training and evaluation parameters.
- `tests/`: CPU tests for the package.
- `examples/` and `data/samples/`: sample inputs.
- `docs/` and `results/`: technical documentation and experiment metrics.
- `notebooks/legacy/`: my original notebooks.

I keep model weights and large datasets outside Git.
