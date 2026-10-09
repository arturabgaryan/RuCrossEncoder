"""Run after training or exporting a complete model artifact."""

import argparse

from rucrossencoder.inference import Reranker

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()
ranker = Reranker(args.model)
for document in ranker.rerank(
    "Как поменять пароль?",
    [
        "Измените пароль в настройках аккаунта.",
        "Сегодня идёт дождь.",
    ],
):
    print(document)
