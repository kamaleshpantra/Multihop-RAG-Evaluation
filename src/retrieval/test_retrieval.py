import json

from src.retrieval.retriever import retrieve


DATA_PATH = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"


def main():

    # Load one HotpotQA example
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    example = data[0]

    question = example["question"]
    answer = example["answer"]
    supporting_facts = example["supporting_facts"]

    print("\n==============================")
    print("HOTPOTQA EXAMPLE")
    print("==============================")

    print("\nQuestion:")
    print(question)

    print("\nGold Answer:")
    print(answer)

    print("\nGold Supporting Facts:")
    for fact in supporting_facts:
        print(fact)

    # Retrieve documents using ONLY the question
    results = retrieve(
        question,
        k=5
    )

    print("\n==============================")
    print("RETRIEVED DOCUMENTS")
    print("==============================")

    for i, document in enumerate(
        results,
        start=1
    ):

        title = document.metadata["title"]

        sentence_ids = document.metadata[
            "sentence_ids"
        ]

        print(
            f"\n---------- RESULT {i} ----------"
        )

        print("Title:")
        print(title)

        print("Sentence IDs:")
        print(sentence_ids)

        print("\nText:")
        print(document.page_content)


if __name__ == "__main__":
    main()