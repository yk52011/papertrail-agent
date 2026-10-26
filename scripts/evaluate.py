"""Synthetic retrieval smoke evaluation, not an end-to-end Agent benchmark."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from papertrail.retrieval import RepositoryIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("eval/results/offline.json"))
    args = parser.parse_args()
    load_dotenv()
    root = Path(__file__).resolve().parents[1]
    embeddings = None
    if args.semantic:
        from papertrail.agent import configured_embeddings
        embeddings = configured_embeddings()
        if embeddings is None:
            parser.error("Set PAPERTRAIL_EMBEDDING_MODEL for semantic evaluation")
    index = RepositoryIndex(root / "eval/corpus", embeddings=embeddings)
    tasks = json.loads((root / "eval/tasks.json").read_text())
    rows = []
    for task in tasks:
        # Deliberately weak, explicit baseline: always supply first indexed chunk.
        baseline = index.chunks[0].source
        hits = index.search(task["query"], k=3, mode="hybrid" if embeddings else "bm25")
        rows.append({"id": task["id"], "expected": task["source"], "baseline": baseline,
                     "retrieved": [h["citation"] for h in hits],
                     "baseline_correct": baseline == task["source"],
                     "top1_correct": bool(hits and hits[0]["source"] == task["source"]),
                     "top3_correct": any(h["source"] == task["source"] for h in hits)})
    corrected = [r["id"] for r in rows if not r["baseline_correct"] and r["top1_correct"]]
    corrupted = [r["id"] for r in rows if r["baseline_correct"] and not r["top1_correct"]]
    result = {"scope": "12 handcrafted English retrieval tasks; no LLM or real-repo claims",
              "mode": "hybrid" if embeddings else "bm25", "n": len(rows),
              "baseline_top1": sum(r["baseline_correct"] for r in rows) / len(rows),
              "retrieval_top1": sum(r["top1_correct"] for r in rows) / len(rows),
              "retrieval_top3": sum(r["top3_correct"] for r in rows) / len(rows),
              "corrected_ids": corrected, "corrupted_ids": corrupted,
              "net_gain_count": len(corrected) - len(corrupted), "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
