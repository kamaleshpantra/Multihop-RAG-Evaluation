def build_prompt(question: str, documents) -> str:
    evidence_parts = []

    for i, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "Unknown")
        text = document.page_content.strip()

        evidence_parts.append(
            f"Evidence {i}:\n"
            f"Source: {title}\n"
            f"Content: {text}"
        )

    evidence = "\n\n".join(evidence_parts)

    prompt = f"""You are a precise question-answering system.

Answer the question using only the evidence provided below.

Follow these rules carefully:
1. Identify the exact people, entities, or objects mentioned in the question.
2. Use information about those exact entities, not similarly named entities.
3. Ignore information about films, places, or other entities unless it directly answers the question.
4. Compare the relevant evidence before deciding the answer.
5. For a yes/no question, answer exactly "yes" or "no".
6. Do not use outside knowledge.
7. Do not explain your answer.
8. Do not mention the evidence.

Question:
{question}

Evidence:
{evidence}

Final answer:"""

    return prompt