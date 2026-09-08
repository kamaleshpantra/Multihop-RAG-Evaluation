from langchain_core.documents import Document

from src.evaluation.retrieval_metrics import (
    supporting_fact_match,
    calculate_recall_at_k,
    calculate_mrr
)


def test_supporting_fact_match():

    document = Document(
        page_content="Edward Wood was an American filmmaker.",
        metadata={
            "title": "Ed Wood",
            "sentence_ids": [0],
        },
    )

    assert supporting_fact_match(
        ["Ed Wood", 0],
        document
    )

    assert not supporting_fact_match(
        ["Ed Wood", 1],
        document
    )

    assert not supporting_fact_match(
        ["Scott Derrickson", 0],
        document
    )


def test_recall_at_k():

    documents = [
        Document(
            page_content="Ed Wood was American.",
            metadata={
                "title": "Ed Wood",
                "sentence_ids": [0],
            },
        )
    ]

    supporting_facts = [
        ["Scott Derrickson", 0],
        ["Ed Wood", 0],
    ]

    recall = calculate_recall_at_k(
        supporting_facts,
        documents
    )

    assert recall == 0.5

def test_mrr():
    documents = [
        Document(
            page_content="Random document",
            metadata={
                "title": "Random",
                "sentence_ids": [0],
            },
        ),
        Document(
            page_content="Ed Wood was American.",
            metadata={
                "title": "Ed Wood",
                "sentence_ids": [0],
            },
        ),
    ]

    supporting_facts = [
        ["Ed Wood", 0],
    ]

    mrr = calculate_mrr(
        supporting_facts,
        documents,
    )

    assert mrr == 0.5