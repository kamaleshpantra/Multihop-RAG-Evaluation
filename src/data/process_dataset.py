import json
from pathlib import Path

from src.data.loader import load_hotpotqa, extract_context
from src.indexing.chunker import chunk_documents


INPUT_PATH = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"
OUTPUT_PATH = "data/processed/chunks.json"


def main():

    print("Loading HotpotQA...")

    data = load_hotpotqa(INPUT_PATH)

    print(f"Loaded {len(data)} examples.")

    print("Extracting documents...")

    documents = extract_context(data)

    print(f"Extracted {len(documents)} documents.")

    print("Creating chunks...")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()