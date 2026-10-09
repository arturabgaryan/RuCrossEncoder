import json

import pytest
import torch

from rucrossencoder.data import write_jsonl
from rucrossencoder.evaluate import evaluate
from rucrossencoder.inference import Reranker
from rucrossencoder.train import train


def test_cpu_train_export_rerank_and_evaluate(tiny_encoder, tmp_path):
    train_path, dev_path = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_jsonl(
        [{"query": f"вопрос {i}", "positive": "ответ", "negative": "другой"} for i in range(3)],
        train_path,
    )
    write_jsonl([{"query": "вопрос dev", "positive": "ответ", "negative": "другой"}], dev_path)
    config = {
        "encoder_id": str(tiny_encoder),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "output_dir": str(tmp_path / "run"),
        "epochs": 1,
        "batch_size": 2,
        "max_length": 16,
        "accumulation_steps": 2,
        "device": "cpu",
    }
    history = train(config)
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["dev_loss"]))
    model_path = tmp_path / "run" / "best"
    assert len(Reranker(str(model_path)).rerank("вопрос", ["ответ", "другой"])) == 2
    metrics = evaluate(
        {
            "test_path": str(dev_path),
            "output_dir": str(tmp_path / "eval"),
            "models": [{"name": "tiny", "model_id": str(model_path)}],
        }
    )
    assert metrics["tiny"]["pairs"] == 2
    assert json.loads((tmp_path / "eval" / "manifest.json").read_text())["test_sha256"]
    # Completed epoch resume is a no-op with unchanged data/config.
    config["resume_from"] = str(tmp_path / "run" / "resume.pt")
    assert train(config) == history
    config["batch_size"] = 3
    with pytest.raises(ValueError, match="unchanged"):
        train(config)


def test_epoch_resume_matches_uninterrupted_run(tiny_encoder, tmp_path, monkeypatch):
    train_path, dev_path = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_jsonl(
        [{"query": f"вопрос {i}", "positive": "ответ", "negative": "другой"} for i in range(3)],
        train_path,
    )
    write_jsonl([{"query": "dev", "positive": "ответ", "negative": "другой"}], dev_path)
    config = {
        "encoder_id": str(tiny_encoder),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "output_dir": str(tmp_path / "full"),
        "epochs": 2,
        "batch_size": 2,
        "max_length": 16,
        "accumulation_steps": 2,
        "device": "cpu",
        "seed": 17,
    }
    expected = train(config)
    save = torch.save

    def interrupt_after_checkpoint(obj, path, *args, **kwargs):
        save(obj, path, *args, **kwargs)
        if str(path).endswith("resume.pt"):
            raise RuntimeError("simulated interruption")

    config["output_dir"] = str(tmp_path / "resumed")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupt_after_checkpoint)
        with pytest.raises(RuntimeError, match="simulated"):
            train(config)
    config["resume_from"] = str(tmp_path / "resumed" / "resume.pt")
    assert train(config) == expected
