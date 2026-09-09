from src.retrieval.recall_preserving_multihop import (
    RecallPreservingMultiHopRetriever,
)


def main():

    print("=" * 70)
    print("RECALL-PRESERVING MULTI-HOP TEST")
    print("=" * 70)

    retriever = RecallPreservingMultiHopRetriever(
        candidate_k=20,
        initial_k=15,
        expansion_k=5,
        expansion_candidate_k=20,
        protected_k=5,
        final_k=15,
    )

    questions = [
        (
            "Were Scott Derrickson and Ed Wood "
            "of the same nationality?"
        ),
        (
            "The arena where the Lewiston Maineiacs "
            "played their home games can seat how "
            "many people?"
        ),
        (
            "Brown State Fishing Lake is in a country "
            "that has a population of how many inhabitants?"
        ),
        (
            "Are Giuseppe Verdi and Ambroise Thomas "
            "both Opera composers?"
        ),
        (
            "When was Poison's album "
            '"Shut Up, Make Love" released?'
        ),
    ]

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(f"QUESTION {index}")
        print("=" * 70)

        print(question)

        documents = retriever.retrieve(
            question,
            k=15,
        )

        print("\nRETRIEVED DOCUMENTS")
        print("-" * 70)

        for rank, document in enumerate(
            documents,
            start=1,
        ):

            title = document.metadata.get(
                "title",
                "Unknown",
            )

            score = document.metadata.get(
                "reranker_score"
            )

            if score is not None:

                print(
                    f"[{rank}] {title} "
                    f"(score={score:.4f})"
                )

            else:

                print(
                    f"[{rank}] {title}"
                )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()