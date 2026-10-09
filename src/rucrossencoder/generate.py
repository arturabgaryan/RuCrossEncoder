"""Resumable JSONL generation. Structural checks do not guarantee factual correctness."""

import re
import time
from pathlib import Path

from .data import query_key, read_records, validate_triples

PROMPT = """Выведи только JSON с тремя строковыми полями query, positive, negative.
На русском языке придумай вопрос на произвольную тему, правильный ответ на него
и нерелевантный ответ на совершенно другой вопрос. Не повторяй примеры."""


def generate(config):
    import json

    import ollama

    output = Path(config.get("output_path", "data/raw/generated.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_records(output) if output.exists() and output.stat().st_size else []
    validate_triples(existing) if existing else None
    seen = {query_key(row["query"]) for row in existing}
    count = int(config.get("count", 200000))
    retries = int(config.get("retries", 3))
    if count < 1 or retries < 1:
        raise ValueError("count and retries must be positive")
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in ("query", "positive", "negative")},
        "required": ["query", "positive", "negative"],
        "additionalProperties": False,
    }
    with output.open("a", encoding="utf-8") as stream:
        for index in range(len(existing), count):
            for attempt in range(retries):
                try:
                    response = ollama.chat(
                        model=config["model_id"],
                        messages=[{"role": "user", "content": PROMPT}],
                        format=schema,
                        options={
                            "temperature": config.get("temperature", 0.7),
                            "top_p": config.get("top_p", 0.9),
                            "seed": int(config.get("seed", 42)) + index * retries + attempt,
                        },
                    )
                    row = json.loads(response.message.content)
                    validate_triples([row])
                    if any(not re.search("[А-Яа-яЁё]", row[key]) for key in schema["required"]):
                        raise ValueError("Each field must contain Cyrillic text")
                    key = query_key(row["query"])
                    if key in seen:
                        raise ValueError("Duplicate query")
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    seen.add(key)
                    break
                except (ValueError, ollama.ResponseError) as exc:
                    if attempt + 1 == retries:
                        raise RuntimeError(
                            f"Generation stopped at record {index}; rerun to resume"
                        ) from exc
                    time.sleep(min(attempt + 1, 3))
            if (index + 1) % 100 == 0:
                print(f"Generated {index + 1}/{count}", flush=True)
