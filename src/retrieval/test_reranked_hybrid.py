from src.retrieval.reranked_hybrid_retriever import (
    RerankedHybridRetriever,
)


def main():

    question = (
        "Were Scott Derrickson and Ed Wood "
        "of the same nationality?"
    )

    print("=" * 60)
    print("BUILDING RERANKED HYBRID RETRIEVER")
    print("=" * 60)

    retriever = RerankedHybridRetriever(
        candidate_k=20
    )

    results = retriever.retrieve(
        question,
        k=10,
    )

    print()
    print("=" * 60)
    print("RERANKED HYBRID RESULTS")
    print("=" * 60)

    for rank, document in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"---------- RANK {rank} ----------")

        print(
            f"Title: "
            f"{document.metadata['title']}"
        )

        print(
            f"Sentence IDs: "
            f"{document.metadata['sentence_ids']}"
        )

        print(
            f"Reranker score: "
            f"{document.metadata['reranker_score']:.4f}"
        )

        print(
            f"Dense rank: "
            f"{document.metadata.get('dense_rank')}"
        )

        print(
            f"BM25 rank: "
            f"{document.metadata.get('bm25_rank')}"
        )

        print()
        print(f"Text:\n{document.page_content}")


if __name__ == "__main__":
    main()