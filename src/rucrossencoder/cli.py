import argparse
import json
from pathlib import Path

from .data import read_records, split_triples, write_jsonl
from .utils import load_config


def main():
    parser = argparse.ArgumentParser(prog="rucrossencoder")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "generate"):
        sub.add_parser(command).add_argument("--config", required=True)
    split = sub.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--seed", type=int, default=42)
    rank = sub.add_parser("rerank")
    rank.add_argument("--model", required=True)
    rank.add_argument("--input", required=True, help="JSON containing query and documents")
    rank.add_argument("--slow-tokenizer", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--encoder-config", required=True)
    export.add_argument("--tokenizer", required=True)
    export.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "split":
        for name, rows in zip(
            ("train", "dev", "test"), split_triples(read_records(args.input), args.seed)
        ):
            write_jsonl(rows, Path(args.output_dir) / f"{name}.jsonl")
    elif args.command == "export":
        from .export import export_checkpoint

        export_checkpoint(args.checkpoint, args.encoder_config, args.tokenizer, args.output_dir)
    elif args.command == "rerank":
        from .inference import Reranker

        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        print(
            json.dumps(
                Reranker(args.model, use_fast=not args.slow_tokenizer).rerank(
                    data["query"], data["documents"]
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        config = load_config(args.config)
        if args.command == "train":
            from .train import train

            train(config)
        elif args.command == "evaluate":
            from .evaluate import evaluate

            evaluate(config)
        else:
            from .generate import generate

            generate(config)


if __name__ == "__main__":
    main()
