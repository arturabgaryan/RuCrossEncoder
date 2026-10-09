"""Convert legacy tensor checkpoints only with an explicit compatible config/tokenizer."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from .model import RuCrossEncoderConfig, RuCrossEncoderModel


def export_checkpoint(checkpoint_path, encoder_config_path, tokenizer_path, output_dir):
    backbone_config = AutoConfig.from_pretrained(encoder_config_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # DeBERTa-v3 reserves embedding rows beyond its tokenizer vocabulary.
    # Equal sizes are unnecessary; every emitted token ID must fit the matrix.
    if max(tokenizer.get_vocab().values()) >= backbone_config.vocab_size:
        raise ValueError("Tokenizer emits IDs outside encoder embeddings")
    config = RuCrossEncoderConfig(
        encoder_config=backbone_config.to_dict(),
        num_labels=2,
        id2label={0: "irrelevant", 1: "relevant"},
        label2id={"irrelevant": 0, "relevant": 1},
    )
    model = RuCrossEncoderModel(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    # No missing/extra keys or silent shape conversions are accepted.
    model.load_state_dict(state, strict=True)
    output = Path(output_dir)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
