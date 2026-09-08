from src.data.loader import load_hotpotqa, extract_context


DATA_PATH = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"


def test_dataset_loads():

    data = load_hotpotqa(DATA_PATH)

    assert len(data) > 0


def test_context_extraction():

    data = load_hotpotqa(DATA_PATH)

    documents = extract_context(data)

    assert len(documents) > 0

    assert "title" in documents[0]
    assert "text" in documents[0]