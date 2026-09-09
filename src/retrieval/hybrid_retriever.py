from src.retrieval.retriever import load_vectorstore
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(
        self,
        dense_retriever=None,
        bm25_retriever=None,
        rrf_k=60,
    ):

        self.dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else load_vectorstore()
        )

        self.bm25_retriever = (
            bm25_retriever
            if bm25_retriever is not None
            else BM25Retriever()
        )

        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 10,
        candidate_k: int = 10,
    ):

        dense_documents = (
            self.dense_retriever.similarity_search(
                query,
                k=candidate_k
            )
        )

        bm25_documents = (
            self.bm25_retriever.retrieve(
                query,
                k=candidate_k
            )
        )

        scores = {}
        documents = {}
        ranks = {}

        self._add_results(
            dense_documents,
            scores,
            documents,
            ranks,
            "dense"
        )

        self._add_results(
            bm25_documents,
            scores,
            documents,
            ranks,
            "bm25"
        )

        ranked_documents = sorted(
            documents.values(),
            key=lambda document: scores[
                self._document_key(document)
            ],
            reverse=True
        )

        for document in ranked_documents:

            key = self._document_key(document)

            document.metadata["rrf_score"] = (
                scores[key]
            )

            document.metadata["dense_rank"] = (
                ranks[key].get("dense")
            )

            document.metadata["bm25_rank"] = (
                ranks[key].get("bm25")
            )

        return ranked_documents[:k]

    def _add_results(
        self,
        retrieved_documents,
        scores,
        documents,
        ranks,
        retriever_name,
    ):

        for rank, document in enumerate(
            retrieved_documents,
            start=1
        ):

            key = self._document_key(document)

            if key not in scores:
                scores[key] = 0.0

            scores[key] += (
                1 / (self.rrf_k + rank)
            )

            documents[key] = document

            if key not in ranks:
                ranks[key] = {}

            ranks[key][retriever_name] = rank

    @staticmethod
    def _document_key(document):

        return (
            document.metadata["title"],
            document.metadata["source_id"],
            document.page_content,
        )