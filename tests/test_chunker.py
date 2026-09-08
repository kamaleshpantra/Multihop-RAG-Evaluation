from src.data.loader import load_hotpotqa, extract_context
from src.indexing.chunker import chunk_documents


DATA_PATH = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"


def test_chunking():

    data = load_hotpotqa(DATA_PATH)

    documents = extract_context(data)

    chunks = chunk_documents(documents)

    assert len(chunks) > 0

    for chunk in chunks[:10]:

        assert "title" in chunk
        assert "text" in chunk
        assert "source_id" in chunk
        assert "sentence_ids" in chunk

        assert len(chunk["text"]) > 0
        assert isinstance(chunk["sentence_ids"], list)