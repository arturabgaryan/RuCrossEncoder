# Legacy checkpoint inspection and conversion

The author retains only
https://huggingface.co/ArturAbg/RuCrossEncoder/blob/main/checkpoint.pt.
A read-only HTTP range inspection parsed its first ZIP entry with pickletools,
without unpickling objects or downloading tensor storage. Observations:

- Total file size: 2,213,369,166 bytes.
- ZIP metadata entry: checkpoint_epoch_9/data.pkl; epoch is 9 (zero-based).
- model_state_dict, optimizer_state_dict, scheduler_state_dict and scaler_state_dict.
- Word embedding dimensions: 128100 x 768.
- Encoder layer indices 0 through 11; DeBERTa-style query/key/value projections.
- Head keys match classifier.fc1, layernorm and fc2 in the original notebook.

These are structural observations, not proof of a particular original model ID.
All 198 encoder tensor names and shapes match microsoft/deberta-v3-base exactly.
Sampled embeddings closely match that pretrained model. Its standard tokenizer
has 128001 entries, while its config reserves 128100 embedding rows; the difference
does not demonstrate custom added tokens. This is strong candidate evidence,
not proof of the original tokenizer's precise settings.

The metadata does not supply an original tokenizer artifact. Vocabulary size alone
cannot recover token-to-ID mapping, normalization, special tokens or added tokens.
Do not substitute a similar-sized tokenizer and present its outputs as verified.
Attention heads, relative-position settings and other config fields also need
original provenance; matching tensor shapes alone does not prove identical behavior.

Once a compatible original config and tokenizer are recovered:

```bash
rucrossencoder export --checkpoint checkpoint.pt --encoder-config path/to/config-directory --tokenizer path/to/tokenizer --output-dir artifacts/legacy-export
```

Conversion uses weights_only=True, checks token IDs fit embeddings and strictly loads all
state keys/shapes. It does not infer the missing config or reconstruct a vocabulary.
The full checkpoint must be available locally, with enough RAM for tensor storage.
An independently prepared candidate export now loads all 204 original model tensors
strictly under the microsoft/deberta-v3-base configuration and standard tokenizer.
All storage CRCs match the checkpoint. A diagnostic run on the first 200 published
generated triples (400 pairs, length 128, threshold 0.5) yielded accuracy 0.6425,
precision 0.72093, recall 0.465, F1 0.56535 and positive-above-negative fraction 0.775.
Fast/slow token IDs matched on those pairs. The historical report is not reproduced;
candidate structural compatibility does not prove exact historical tokenization or
configuration. Weights remain external artifacts, not files committed to this repo.
