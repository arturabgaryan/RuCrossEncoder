# Architecture

The package preserves the original joint encoder and head:
Linear(hidden, hidden), GELU, LayerNorm, Dropout(0.1), Linear(hidden, 2).
Pooling uses `pooler_output` when present, otherwise the first token embedding.
Two-label training uses cross entropy; relevance is softmax(logits)[:, 1].
The regression branch uses MSE, but the supplied training pipeline trains
two-label classification only.

RuCrossEncoderConfig stores the backbone configuration and head settings.
RuCrossEncoderModel.save_pretrained() exports backbone and head as safetensors
with config.json. Save the tokenizer in the same directory. Load using the
package's RuCrossEncoderModel, not the generic AutoModel class. No custom
Auto-class registration or remote code execution is required.

The exact historical DeBERTa ID/revision and tokenizer are unresolved. No new
morpheme tokenizer or domain pretraining is claimed by this refactor.
Changing max_position_embeddings alone does not extend context. Sample configs
retain the original effective maximum of 128 input tokens.
