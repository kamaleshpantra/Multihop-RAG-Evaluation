from src.generation.rag_pipeline import RAGPipeline


def main():

    question = "Were Scott Derrickson and Ed Wood of the same nationality?"

    print("=" * 70)
    print("INITIALIZING RAG PIPELINE")
    print("=" * 70)

    # Use the normal RAG configuration.
    #
    # candidate_k=20:
    #   Retrieve 20 candidates from the hybrid retriever
    #   before reranking.
    #
    # retrieval_k=10:
    #   Pass the top 10 reranked documents to Qwen.
    pipeline = RAGPipeline(
        candidate_k=20,
        retrieval_k=10
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # Run the complete RAG pipeline:
    #
    # Question
    #     ↓
    # Dense + BM25
    #     ↓
    # RRF
    #     ↓
    # Cross-Encoder Reranker
    #     ↓
    # Top-10 evidence
    #     ↓
    # Qwen 2.5 1.5B
    #     ↓
    # Answer
    result = pipeline.answer(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for i, document in enumerate(result["documents"], start=1):

        title = document.metadata.get(
            "title",
            "Unknown"
        )

        print(f"\n[{i}] {title}")

        print(document.page_content[:500])

    print("\n" + "=" * 70)
    print("RAG PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()