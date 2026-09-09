from src.generation.llm import generate_answer


def main():

    prompt = """You are answering a question using the provided evidence.

Question:
Were Scott Derrickson and Ed Wood of the same nationality?

Evidence:
Scott Derrickson was an American director, screenwriter and producer.

Ed Wood was an American filmmaker, actor, writer, producer and director.

Answer only with:
yes
or
no

Answer:"""

    answer = generate_answer(prompt)

    print("=" * 70)
    print("LLM TEST")
    print("=" * 70)

    print("\nQuestion:")
    print("Were Scott Derrickson and Ed Wood of the same nationality?")

    print("\nLLM Answer:")
    print(answer)


if __name__ == "__main__":
    main()