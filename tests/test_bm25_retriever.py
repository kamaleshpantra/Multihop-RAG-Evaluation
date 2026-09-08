from src.retrieval.bm25_retriever import tokenize


def test_tokenize():

    text = "Ed Wood was an American filmmaker."

    tokens = tokenize(text)

    assert tokens == [
        "ed",
        "wood",
        "was",
        "an",
        "american",
        "filmmaker",
    ]


def test_tokenize_lowercase():

    assert tokenize("AMERICAN Filmmaker") == [
        "american",
        "filmmaker",
    ]


def test_tokenize_punctuation():

    assert tokenize("Ed Wood: director, writer!") == [
        "ed",
        "wood",
        "director",
        "writer",
    ]