from src.generation.llm import generate_answer


def run_test(name, evidence):
    prompt = f"""You are a precise question-answering system.

Answer the question using ONLY the evidence.

Question:
Were Scott Derrickson and Ed Wood of the same nationality?

Evidence:
{evidence}

Answer only "yes" or "no".

Answer:"""

    answer = generate_answer(prompt)

    print("=" * 70)
    print(name)
    print("=" * 70)
    print("Answer:", answer)


def main():

    run_test(
        "TEST 1 - Scott + Ed Wood",
        """
Scott Derrickson was an American director.

Edward Davis Wood Jr. was an American filmmaker.
"""
    )

    run_test(
        "TEST 2 - Ed Wood only",
        """
Edward Davis Wood Jr. was an American filmmaker, actor,
writer, producer and director.
"""
    )

    run_test(
        "TEST 3 - Scott only",
        """
Scott Derrickson was an American director, screenwriter
and producer.
"""
    )


if __name__ == "__main__":
    main()