from src.retrieval.multihop_retriever import MultiHopRetriever


def main():

    print("=" * 70)
    print("MULTI-HOP RETRIEVAL TEST")
    print("=" * 70)

    retriever = MultiHopRetriever(
        candidate_k=20,
        initial_k=10,
        expansion_k=5,
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
            "The football manager who recruited David "
            "Beckham managed Manchester United during "
            "what timeframe?"
        ),
        (
            "What science fantasy young adult series, "
            "told in first person, has a set of companion "
            "books narrating the stories of enslaved "
            "worlds and alien species?"
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

            print(
                f"[{rank}] {title}"
            )

            if score is not None:
                print(
                    f"     score: {score:.4f}"
                )

    print("\n" + "=" * 70)
    print("MULTI-HOP RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()