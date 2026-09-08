from src.indexing.embeddings import get_embedding_model


def test_embedding_model():

    embeddings = get_embedding_model()

    vector = embeddings.embed_query(
        "Who founded Apple?"
    )

    assert isinstance(vector, list)

    assert len(vector) == 384

    assert all(
        isinstance(value, float)
        for value in vector
    )

    print("\nEmbedding dimension:", len(vector))