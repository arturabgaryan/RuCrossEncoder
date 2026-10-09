import pytest
import torch

from rucrossencoder.evaluate import classification_metrics, ranking_metrics
from rucrossencoder.inference import relevance_scores


def test_two_logits_produce_one_score_per_document():
    scores = relevance_scores(torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    with pytest.raises(ValueError):
        relevance_scores(torch.ones(3, 2), score_mode="sigmoid")
    assert relevance_scores(torch.zeros(2, 1), score_mode="sigmoid").tolist() == [0.5, 0.5]


def test_classification_metrics_and_empty_guard():
    result = classification_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
    assert result["f1"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    with pytest.raises(ValueError):
        classification_metrics([], [], 0.5)


def test_ranking_known_order():
    perfect = ranking_metrics([0, 2, 1], [0.1, 0.9, 0.8], 3)
    assert perfect["ndcg@3"] == pytest.approx(1.0)
    assert perfect["mrr@3"] == 1.0
    assert perfect["map@3"] == 1.0
    imperfect = ranking_metrics([0, 1], [0.9, 0.1], 2)
    assert imperfect["mrr@2"] == 0.5
    assert imperfect["ndcg@2"] < 1.0
    with pytest.raises(ValueError):
        ranking_metrics([], [], 10)
