"""Training with dynamic padding, accumulation, dev evaluation and epoch resume."""

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import PairCollator, PairDataset, assert_disjoint, read_records
from .model import RuCrossEncoderModel
from .utils import manifest, set_seed, write_json


def train(config):
    encoder = config.get("encoder_id")
    if not encoder:
        raise ValueError("Set encoder_id to the actual backbone used in your experiment")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    output = Path(config.get("output_dir", "artifacts/training"))
    output.mkdir(parents=True, exist_ok=True)
    train_rows = read_records(config["train_path"])
    dev_rows = read_records(config["dev_path"])
    assert_disjoint(train_rows, dev_rows)
    device_name = config.get("device", "auto")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    revision = config.get("encoder_revision")
    tokenizer_id = config.get("tokenizer_id") or encoder
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=config.get("tokenizer_revision") or revision,
    )
    collator = PairCollator(tokenizer, int(config.get("max_length", 128)))
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config.get("batch_size", 32))
    loader = DataLoader(
        PairDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    dev_loader = DataLoader(PairDataset(dev_rows), batch_size=batch_size, collate_fn=collator)
    model = RuCrossEncoderModel.from_encoder_pretrained(
        encoder,
        revision=revision,
        dropout_prob=float(config.get("dropout_prob", 0.1)),
    ).to(device)
    epochs = int(config.get("epochs", 10))
    accumulation = int(config.get("accumulation_steps", 1))
    if epochs < 1 or accumulation < 1:
        raise ValueError("epochs and accumulation_steps must be positive")
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        eps=1e-6,
    )
    steps = math.ceil(len(loader) / accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(float(config.get("warmup_ratio", 0.1)) * steps),
        steps,
    )
    amp = bool(config.get("fp16", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_loss, history = 0, float("inf"), []
    fingerprints = {
        name: hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
        for name in ("train_path", "dev_path")
    }
    contract = {k: v for k, v in config.items() if k not in {"resume_from", "output_dir"}}
    if config.get("resume_from"):
        # Only load a checkpoint produced locally by this training code.
        checkpoint = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
        if checkpoint["contract"] != contract or checkpoint["fingerprints"] != fingerprints:
            raise ValueError("Resume requires unchanged training configuration and data")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss, history = checkpoint["best_loss"], checkpoint["history"]
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        generator.set_state(checkpoint["loader_state"])
        if device.type == "cuda" and checkpoint["cuda_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
    run_manifest = manifest(config)
    run_manifest["data_sha256"] = fingerprints
    run_manifest["resolved_encoder_revision"] = getattr(model.encoder.config, "_commit_hash", None)
    write_json(run_manifest, output / "manifest.json")
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Weight by samples, including a potentially short final microbatch.
            group_start = (step // accumulation) * accumulation
            group_samples = min(
                accumulation * batch_size,
                len(loader.dataset) - group_start * batch_size,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            train_loss += loss.detach().item() * len(batch["labels"])
            scaler.scale(loss * len(batch["labels"]) / group_samples).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        dev_loss = 0.0
        with torch.inference_mode():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                dev_loss += model(**batch).loss.item() * len(batch["labels"])
        dev_loss /= len(dev_loader.dataset)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(loader.dataset),
            "dev_loss": dev_loss,
        }
        history.append(metrics)
        print(metrics, flush=True)
        if dev_loss < best_loss:
            best_loss = dev_loss
            model.save_pretrained(output / "best", safe_serialization=True)
            tokenizer.save_pretrained(output / "best")
        model.save_pretrained(output / "last", safe_serialization=True)
        tokenizer.save_pretrained(output / "last")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": best_loss,
                "history": history,
                "contract": contract,
                "fingerprints": fingerprints,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "loader_state": generator.get_state(),
                "cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
            output / "resume.pt",
        )
        write_json(history, output / "history.json")
    return history
