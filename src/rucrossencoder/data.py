"""Validated local data loading and query-level splitting."""

import csv
import json
import random
from pathlib import Path

from torch.utils.data import Dataset


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
    elif path.suffix in {".xlsx", ".parquet"}:
        import pandas as pd

        frame = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_parquet(path)
        records = frame.to_dict("records")
    else:
        raise ValueError("Supported formats: .jsonl, .csv, .xlsx, .parquet")
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def validate_triples(records):
    for i, record in enumerate(records):
        for field in ("query", "positive", "negative"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Record {i}: {field} must be a nonempty string")
        if query_key(record["positive"]) == query_key(record["negative"]):
            raise ValueError(f"Record {i}: positive and negative are identical")
    if not records:
        raise ValueError("Empty triple dataset")
    return records


def query_key(text):
    return " ".join(text.casefold().split())


def assert_disjoint(*splits):
    seen = set()
    for records in splits:
        keys = {query_key(row["query"]) for row in records}
        if seen & keys:
            raise ValueError("Query overlap between dataset splits")
        seen.update(keys)


def split_triples(records, seed=42):
    """Group identical normalized queries before an approximately 80/10/10 split."""
    validate_triples(records)
    groups = {}
    for row in records:
        groups.setdefault(query_key(row["query"]), []).append(row)
    keys = list(groups)
    if len(keys) < 10:
        raise ValueError("At least 10 distinct queries are required for 80/10/10 splitting")
    random.Random(seed).shuffle(keys)
    end_train, end_dev = int(len(keys) * 0.8), int(len(keys) * 0.9)
    splits = [
        [row for key in subset for row in groups[key]]
        for subset in (keys[:end_train], keys[end_train:end_dev], keys[end_dev:])
    ]
    assert_disjoint(*splits)
    return splits


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class PairDataset(Dataset):
    def __init__(self, triples):
        self.triples = validate_triples(triples)

    def __len__(self):
        return len(self.triples) * 2

    def __getitem__(self, idx):
        row = self.triples[idx // 2]
        label = 1 if idx % 2 == 0 else 0
        return row["query"], row["positive" if label else "negative"], label


class PairCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        import torch

        queries, texts, labels = zip(*rows)
        batch = self.tokenizer(
            list(queries),
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
