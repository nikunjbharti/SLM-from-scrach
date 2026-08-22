from datasets import load_dataset
import json

LIMIT = 100 * 1024 * 1024  # 100 MB
output_file = "tinystories_100mb.jsonl"

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

total_size = 0
count = 0

with open(output_file, "w", encoding="utf-8") as f:
    for example in dataset:
        line = json.dumps(example, ensure_ascii=False) + "\n"
        size = len(line.encode("utf-8"))

        if total_size + size > LIMIT:
            break

        f.write(line)

        total_size += size
        count += 1

        if count % 1000 == 0:
            print(
                f"Examples: {count:,} | "
                f"Size: {total_size / 1024**2:.2f} MB"
            )

print(f"\nDone!")
print(f"Examples: {count:,}")
print(f"Size: {total_size / 1024**2:.2f} MB")
print(f"Saved to: {output_file}")