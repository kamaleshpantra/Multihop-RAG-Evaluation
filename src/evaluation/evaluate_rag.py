import argparse
import json
import re
import time
from pathlib import Path

from src.data.loader import load_hotpotqa
from src.generation.rag_pipeline import RAGPipeline


def normalize_answer(answer):
    answer = answer.lower().strip()

    answer = re.sub(r"[^a-z0-9\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer)

    return answer


def token_f1(prediction, gold):
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = set(prediction_tokens) & set(gold_tokens)

    if not common:
        return 0.0

    common_count = sum(
        min(prediction_tokens.count(token), gold_tokens.count(token))
        for token in common
    )

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def evaluate(args):
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    dataset_path = Path(
        "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"
    )

    data = load_hotpotqa(dataset_path)

    if args.limit is not None:
        data = data[:args.limit]

    print(f"Loaded {len(data):,} examples.")

    print()
    print("=" * 70)
    print("BUILDING RAG PIPELINE")
    print("=" * 70)

    pipeline = RAGPipeline(
        candidate_k=args.candidate_k,
        retrieval_k=args.retrieval_k,
        initial_k=args.initial_k,
        expansion_k=args.expansion_k,
        expansion_candidate_k=args.expansion_candidate_k,
        protected_k=args.protected_k,
    )

    predictions = []

    exact_matches = 0
    f1_scores = []

    start_time = time.time()

    print()
    print("=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    for i, example in enumerate(data, start=1):
        question = example["question"]
        gold_answer = example["answer"]

        question_start = time.time()

        result = pipeline.answer(question)

        predicted_answer = result["answer"]

        normalized_prediction = normalize_answer(predicted_answer)
        normalized_gold = normalize_answer(gold_answer)

        exact_match = (
            normalized_prediction == normalized_gold
        )

        f1 = token_f1(
            predicted_answer,
            gold_answer,
        )

        if exact_match:
            exact_matches += 1

        f1_scores.append(f1)

        retrieved_documents = []

        for document in result["documents"]:
            retrieved_documents.append(
                {
                    "title": document.metadata.get(
                        "title",
                        "",
                    ),
                    "text": document.page_content,
                    "source_id": document.metadata.get(
                        "source_id",
                        "",
                    ),
                    "sentence_ids": document.metadata.get(
                        "sentence_ids",
                        [],
                    ),
                    "reranker_score": document.metadata.get(
                        "reranker_score",
                        None,
                    ),
                }
            )

        predictions.append(
            {
                "id": example["_id"],
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "exact_match": exact_match,
                "f1": f1,
                "supporting_facts": example[
                    "supporting_facts"
                ],
                "retrieved_documents": retrieved_documents,
            }
        )

        elapsed = time.time() - question_start

        print(
            f"[{i:>3}/{len(data)}] "
            f"EM={int(exact_match)} "
            f"F1={f1:.4f} "
            f"time={elapsed:.1f}s"
        )

        print(
            f"      Q: {question}"
        )

        print(
            f"      Gold: {gold_answer}"
        )

        print(
            f"      Pred: {predicted_answer}"
        )

    total_time = time.time() - start_time

    exact_match_percentage = (
        exact_matches / len(data) * 100
        if data
        else 0.0
    )

    average_f1 = (
        sum(f1_scores) / len(f1_scores)
        if f1_scores
        else 0.0
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Examples: {len(data)}"
    )

    print(
        f"Exact Match: "
        f"{exact_match_percentage:.2f}%"
    )

    print(
        f"F1: "
        f"{average_f1:.4f}"
    )

    print(
        f"Total time: "
        f"{total_time / 60:.2f} minutes"
    )

    if data:
        print(
            f"Average time/question: "
            f"{total_time / len(data):.2f} seconds"
        )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "configuration": {
            "retriever": (
                "recall_preserving_multihop"
            ),
            "candidate_k": args.candidate_k,
            "retrieval_k": args.retrieval_k,
            "initial_k": args.initial_k,
            "expansion_k": args.expansion_k,
            "expansion_candidate_k": (
                args.expansion_candidate_k
            ),
            "protected_k": args.protected_k,
            "llm": "qwen2.5-coder:7b",
            "temperature": 0.0,
            "seed": 42,
        },
        "metrics": {
            "exact_match": exact_match_percentage,
            "f1": average_f1,
            "examples": len(data),
            "total_time_seconds": total_time,
        },
        "predictions": predictions,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved results to: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--initial-k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--expansion-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--expansion-candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--protected-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "experiments/generation/"
            "rag_multihop_predictions.json"
        ),
    )

    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()