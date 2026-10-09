import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast


@pytest.fixture
def tiny_encoder(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "backbone"
    path.mkdir()
    (path / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nвопрос\nответ\nдругой\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    BertModel(
        BertConfig(
            vocab_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    ).save_pretrained(path)
    return path
