from src.retrieval.recall_preserving_multihop import RecallPreservingMultiHopRetriever
from src.generation.llm import generate_answer
from src.generation.prompt import build_prompt


class RAGPipeline:
    def __init__(
        self,
        candidate_k=20,
        retrieval_k=15,
        initial_k=15,
        expansion_k=5,
        expansion_candidate_k=20,
        protected_k=5,
    ):
        self.retriever = RecallPreservingMultiHopRetriever(
            candidate_k=candidate_k,
            initial_k=initial_k,
            expansion_k=expansion_k,
            expansion_candidate_k=expansion_candidate_k,
            protected_k=protected_k,
            final_k=retrieval_k,
        )

        self.retrieval_k = retrieval_k

    def answer(self, question: str):
        documents = self.retriever.retrieve(
            question,
            k=self.retrieval_k,
        )

        prompt = build_prompt(
            question,
            documents,
        )

        answer = generate_answer(prompt)

        return {
            "question": question,
            "answer": answer,
            "documents": documents,
            "prompt": prompt,
        }