from datasets import load_dataset
import json

def save_haiku_subset(out_file: str = "haiku_subset.jsonl",
                      sample_size: int = 3000,
                      seed: int = 42) -> None:
    """
    Download the statworx/haiku dataset and save a random subset
    as newline-delimited JSON:  {"text": "..."} per line.

    Args:
        out_file: name of the output file.
        sample_size: number of haiku to include.
        seed: random seed for reproducibility.
    """
    ds = load_dataset("statworx/haiku", split="train")
    subset = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))

    with open(out_file, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")

    print(f"Saved {len(subset)} haiku to {out_file}")
save_haiku_subset()