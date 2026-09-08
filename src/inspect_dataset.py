import json
from pathlib import Path

DATA_PATH = Path(
    "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"
)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('Number of examples: ',len(data))

example=data[0]

print('\nKeys:')
print(example.keys())

print('\nQuestion:')
print(example['question'])

print('\nAnswer')
print(example['answer'])

print("\nType:")
print(example["type"])

print("\nLevel:")
print(example["level"])

print("\nSupporting facts:")
print(example["supporting_facts"])

print("\nContext:")
for title, sentences in example["context"]:
    print(f"\n--- {title} ---")

    for i, sentence in enumerate(sentences):
        print(f"{i}: {sentence}")