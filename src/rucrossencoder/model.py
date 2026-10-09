"""Serializable version of the classification architecture in the original notebook."""

from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RuCrossEncoderConfig(PretrainedConfig):
    model_type = "rucrossencoder"

    def __init__(self, encoder_config=None, dropout_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
        self.dropout_prob = dropout_prob


class RuCrossEncoderModel(PreTrainedModel):
    config_class = RuCrossEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        if not config.encoder_config:
            raise ValueError("encoder_config is required; use from_encoder_pretrained()")
        encoder_config = dict(config.encoder_config)
        model_type = encoder_config.pop("model_type")
        backbone_config = AutoConfig.for_model(model_type, **encoder_config)
        self.encoder = AutoModel.from_config(backbone_config)
        size = backbone_config.hidden_size
        # Names match the original notebook's state_dict keys.
        self.classifier = Classifier(size, config.num_labels, config.dropout_prob)

    @classmethod
    def from_encoder_pretrained(cls, name, *, revision=None, dropout_prob=0.1):
        encoder = AutoModel.from_pretrained(name, revision=revision)
        config = RuCrossEncoderConfig(
            encoder_config=encoder.config.to_dict(),
            dropout_prob=dropout_prob,
            num_labels=2,
            id2label={0: "irrelevant", 1: "relevant"},
            label2id={"irrelevant": 0, "relevant": 1},
        )
        model = cls(config)
        model.encoder = encoder
        return model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss = nn.functional.mse_loss(logits.squeeze(-1), labels.float())
            else:
                loss = nn.functional.cross_entropy(logits, labels.long())
        return SequenceClassifierOutput(loss=loss, logits=logits)


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.fc2(self.dropout(self.layernorm(self.act(self.fc1(x)))))
