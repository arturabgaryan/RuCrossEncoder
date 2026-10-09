import pytest

from rucrossencoder.data import PairDataset, assert_disjoint, read_records, split_triples


def rows(n=10):
    return [{"query": f"вопрос {i}", "positive": "ответ", "negative": "другой"} for i in range(n)]


def test_pair_labels_and_mapping():
    dataset = PairDataset(rows(2))
    assert len(dataset) == 4
    assert dataset[0] == ("вопрос 0", "ответ", 1)
    assert dataset[1] == ("вопрос 0", "другой", 0)
    assert dataset[2] == ("вопрос 1", "ответ", 1)


def test_grouped_split_is_repeatable_and_disjoint():
    records = rows(20) + [{"query": " ВОПРОС 0 ", "positive": "ответ 2", "negative": "другой 2"}]
    splits = split_triples(records)
    assert splits == split_triples(records)
    assert sum(map(len, splits)) == len(records)
    assert_disjoint(*splits)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(rows(1), [{"query": " ВОПРОС 0 "}])


def test_reject_invalid_or_empty_data(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty"):
        read_records(path)
    with pytest.raises(ValueError):
        PairDataset([{"query": "q", "positive": "", "negative": "n"}])
