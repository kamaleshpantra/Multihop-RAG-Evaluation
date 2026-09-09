from src.retrieval.reranker import Reranker
from langchain_core.documents import Document


def main():

    query = (
        "Were Scott Derrickson and Ed Wood "
        "of the same nationality?"
    )

    documents = [
        Document(
            page_content=(
                "Ed Wood was an American filmmaker, "
                "actor, writer, and producer."
            ),
            metadata={
                "title": "Ed Wood"
            },
        ),
        Document(
            page_content=(
                "Doctor Strange is a 2016 American "
                "superhero film directed by Scott Derrickson."
            ),
            metadata={
                "title": "Doctor Strange"
            },
        ),
        Document(
            page_content=(
                "The Exorcism of Emily Rose is a 2005 "
                "film directed by Scott Derrickson."
            ),
            metadata={
                "title": "The Exorcism of Emily Rose"
            },
        ),
    ]

    reranker = Reranker()

    results = reranker.rerank(
        query,
        documents,
        k=3,
    )

    print()
    print("=" * 60)
    print("RERANKER RESULTS")
    print("=" * 60)

    for rank, document in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"Rank {rank}")
        print(f"Title: {document.metadata['title']}")
        print(
            f"Score: "
            f"{document.metadata['reranker_score']:.4f}"
        )
        print(
            f"Text: "
            f"{document.page_content}"
        )


if __name__ == "__main__":
    main()