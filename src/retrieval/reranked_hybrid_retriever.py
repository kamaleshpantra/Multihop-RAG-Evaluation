from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


class RerankedHybridRetriever:

    def __init__(
        self,
        hybrid_retriever=None,
        reranker=None,
        candidate_k=20,
    ):

        self.hybrid_retriever = (
            hybrid_retriever
            if hybrid_retriever is not None
            else HybridRetriever(rrf_k=60)
        )

        self.reranker = (
            reranker
            if reranker is not None
            else Reranker()
        )

        self.candidate_k = candidate_k

    def retrieve(
        self,
        query,
        k=10,
    ):

        candidates = self.hybrid_retriever.retrieve(
            query,
            k=self.candidate_k,
            candidate_k=self.candidate_k,
        )

        reranked_documents = self.reranker.rerank(
            query,
            candidates,
            k=k,
        )

        return reranked_documents