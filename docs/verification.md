# Local verification

Date: 2026-10-09. Windows, Python 3.12.14, PyTorch 2.14.1+cpu,
Transformers 4.57.6. Verification uses generated tiny local test models.

Checks:
- 12 pytest tests: data validity, query-disjoint splits, pair labels, score shape,
  classification/ranking metrics, full model serialization, legacy tensor export,
  DeBERTa serialization, reserved embedding rows, CPU training and epoch resume.
- Ruff lint and formatting.
- Dependency lock consistency and wheel packaging.
- Original legacy notebook bytes preserved.

Additional recovery check: all 204 model tensors extracted from the historical
checkpoint with matching CRCs, strict candidate loading and diagnostic inference
on 400 published pairs. Optimizer storages were not loaded.

Not verified: CUDA training, a running Ollama service, exact historical configuration,
published benchmark reproduction or GitHub-hosted CI execution.
