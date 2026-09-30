"""Оценка качества RAG: hit-rate@k, MRR, faithfulness (упрощ.)."""
import json, requests, statistics
from pathlib import Path

API = "http://localhost:8000"
TEST = Path("eval/testset.jsonl")

def hit_at_k(hits, gold_file, k):
    return any(gold_file in h["file"] for h in hits[:k])

def mrr(hits, gold_file):
    for i, h in enumerate(hits):
        if gold_file in h["file"]:
            return 1 / (i + 1)
    return 0.0

def main():
    hits1, mrrs = [], []
    for line in TEST.read_text().splitlines():
        item = json.loads(line)
        r = requests.post(f"{API}/search", json={
            "q": item["q"], "top_k": 20, "rerank": 5, "generate": False,
        }).json()
        hits1.append(hit_at_k(r["hits"], item["gold_file"], 5))
        mrrs.append(mrr(r["hits"], item["gold_file"]))

    print(f"Hit@5: {statistics.mean(hits1):.3f}")
    print(f"MRR:   {statistics.mean(mrrs):.3f}")

if __name__ == "__main__":
    main()
