from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(self, model_name=MODEL_NAME):

        print("Loading reranker model...")

        self.model = CrossEncoder(
            model_name
        )

        print("Reranker model ready.")

    def rerank(
        self,
        query,
        documents,
        k=10,
    ):
        if not documents:
            return []

        pairs = [
            (
                query,
                document.page_content,
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked_documents = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for document, score in ranked_documents[:k]:

            document.metadata[
                "reranker_score"
            ] = float(score)

            results.append(document)

        return results