import torch

from rucrossencoder.export import export_checkpoint
from rucrossencoder.model import RuCrossEncoderModel


def test_save_load_preserves_logits_and_head(tiny_encoder, tmp_path):
    model = RuCrossEncoderModel.from_encoder_pretrained(str(tiny_encoder)).eval()
    inputs = {
        "input_ids": torch.tensor([[2, 5, 3, 6, 3]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    with torch.inference_mode():
        expected = model(**inputs).logits
    directory = tmp_path / "export"
    model.save_pretrained(directory, safe_serialization=True)
    restored = RuCrossEncoderModel.from_pretrained(directory).eval()
    with torch.inference_mode():
        actual = restored(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (directory / "config.json").exists()
    assert (directory / "model.safetensors").exists()


def test_legacy_state_dict_conversion(tiny_encoder, tmp_path):
    model = RuCrossEncoderModel.from_encoder_pretrained(str(tiny_encoder)).eval()
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"epoch": 9, "model_state_dict": model.state_dict()}, checkpoint)
    output = tmp_path / "converted"
    export_checkpoint(checkpoint, tiny_encoder, tiny_encoder, output)
    restored = RuCrossEncoderModel.from_pretrained(output)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_deberta_backbone_roundtrip(tmp_path):
    from transformers import DebertaV2Config

    from rucrossencoder.model import RuCrossEncoderConfig

    config = RuCrossEncoderConfig(
        encoder_config=DebertaV2Config(
            vocab_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
        ).to_dict(),
        num_labels=2,
    )
    model = RuCrossEncoderModel(config).eval()
    inputs = {
        "input_ids": torch.tensor([[2, 5, 3, 6, 3]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    model.save_pretrained(tmp_path / "deberta")
    restored = RuCrossEncoderModel.from_pretrained(tmp_path / "deberta").eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            model(**inputs).logits, restored(**inputs).logits, rtol=0, atol=0
        )


def test_export_accepts_reserved_embedding_rows(tiny_encoder, tmp_path):
    from transformers import AutoConfig

    from rucrossencoder.model import RuCrossEncoderConfig

    backbone = AutoConfig.from_pretrained(tiny_encoder)
    backbone.vocab_size = 10  # Tokenizer has eight entries, with two reserved rows.
    config_path = tmp_path / "config"
    backbone.save_pretrained(config_path)
    model = RuCrossEncoderModel(
        RuCrossEncoderConfig(
            encoder_config=backbone.to_dict(),
            num_labels=2,
        )
    )
    checkpoint = tmp_path / "reserved.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    export_checkpoint(checkpoint, config_path, tiny_encoder, tmp_path / "reserved-export")
