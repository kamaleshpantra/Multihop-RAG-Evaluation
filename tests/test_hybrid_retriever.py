from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever


class FakeDenseRetriever:

    def similarity_search(self, query, k):

        return [
            Document(
                page_content="Document A",
                metadata={
                    "title": "A",
                    "source_id": "1",
                },
            ),
            Document(
                page_content="Document B",
                metadata={
                    "title": "B",
                    "source_id": "2",
                },
            ),
            Document(
                page_content="Document C",
                metadata={
                    "title": "C",
                    "source_id": "3",
                },
            ),
        ][:k]


class FakeBM25Retriever:

    def retrieve(self, query, k):

        return [
            Document(
                page_content="Document B",
                metadata={
                    "title": "B",
                    "source_id": "2",
                },
            ),
            Document(
                page_content="Document A",
                metadata={
                    "title": "A",
                    "source_id": "1",
                },
            ),
            Document(
                page_content="Document D",
                metadata={
                    "title": "D",
                    "source_id": "4",
                },
            ),
        ][:k]


def test_hybrid_retrieval():

    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        bm25_retriever=FakeBM25Retriever(),
        rrf_k=60,
    )

    results = retriever.retrieve(
        "test query",
        k=4,
        candidate_k=3,
    )

    titles = [
        document.metadata["title"]
        for document in results
    ]

    assert titles[0] == "A"
    assert titles[1] == "B"