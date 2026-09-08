def supporting_fact_match(
    supporting_fact,
    document
):
    """
    Check whether a retrieved document
    contains a specific HotpotQA supporting fact.

    supporting_fact:
        [title, sentence_id]

    document:
        LangChain Document
    """

    title, sentence_id = supporting_fact

    retrieved_title = document.metadata["title"]

    retrieved_sentence_ids = document.metadata[
        "sentence_ids"
    ]

    return (
        title == retrieved_title
        and sentence_id in retrieved_sentence_ids
    )


def calculate_recall_at_k(
    supporting_facts,
    retrieved_documents
):
    """
    Calculate evidence recall.

    Example:

    Gold facts = 2
    Retrieved gold facts = 1

    Recall = 1 / 2 = 0.5
    """

    if not supporting_facts:
        return 0.0

    retrieved_count = 0

    for supporting_fact in supporting_facts:

        found = any(
            supporting_fact_match(
                supporting_fact,
                document
            )
            for document in retrieved_documents
        )

        if found:
            retrieved_count += 1

    return retrieved_count / len(supporting_facts)

def calculate_mrr(supporting_facts, retrieved_documents):
    if not supporting_facts:
        return 0.0

    reciprocal_ranks = []

    for supporting_fact in supporting_facts:

        for rank, document in enumerate(retrieved_documents, start=1):

            if supporting_fact_match(supporting_fact, document):
                reciprocal_ranks.append(1 / rank)
                break

        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)