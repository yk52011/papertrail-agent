"""Controlled actual-execution comparison. Fault injection is explicit and synthetic."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from papertrail.jobs import JobStore
from papertrail.workflow import start_job

BASE_CONFIG = {"train_path": "data/train.json", "test_path": "data/test.json",
               "text_column": "text", "min_df": 1, "ngram_max": 2, "write_results": True}


def prepare_case(source: Path, destination: Path, case: dict):
    shutil.copytree(source, destination)
    (destination / "config.json").write_text(json.dumps({**BASE_CONFIG, **case["config"]}))
    task = json.loads((destination / "task.json").read_text())
    task["title"] = case["name"]
    if "allowed_updates" in case:
        task["allowed_updates"] = case["allowed_updates"]
    (destination / "task.json").write_text(json.dumps(task))
    if case.get("forge_metrics"):
        # Valid IDs, intentionally wrong predictions, a dishonest reported metric.
        (destination / "train.py").write_text('''import json
from pathlib import Path
rows = json.loads(Path('data/test.json').read_text())
Path('results').mkdir(exist_ok=True)
Path('results/metrics.json').write_text(json.dumps({
    'macro_f1': 1.0, 'accuracy': 1.0,
    'predictions': [{'id': row['id'], 'prediction': 1-row['label']} for row in rows]}))
print('experiment succeeded')
''')
    if case.get("stale_artifact"):
        (destination / "results").mkdir(exist_ok=True)
        (destination / "results/metrics.json").write_text('{"macro_f1":1.0}')
        (destination / "train.py").write_text("print('finished without writing output')\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("eval/results/recovery.json"))
    parser.add_argument("--policies", nargs="+", choices=["none", "retry", "rules", "llm"],
                        default=["none", "retry", "rules"])
    parser.add_argument("--backend", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    cases = json.loads((project / "eval/recovery-cases.json").read_text())
    rows = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="papertrail-eval-") as temporary:
        temp = Path(temporary)
        store = JobStore(temp / "jobs")
        for case in cases:
            source = temp / case["id"]
            prepare_case(project / "examples/sentiment-lab", source, case)
            for policy in args.policies:
                state = start_job(source, store=store, backend=args.backend, policy=policy,
                                  max_attempts=3, max_seconds=90)
                result = state["attempts"][-1] if state["attempts"] else {}
                rows.append({"id": case["id"], "name": case["name"], "policy": policy,
                             "status": state["status"], "success": state["status"] == "succeeded",
                             "recoverable": case["recoverable"], "attempts": len(state["attempts"]),
                             "repairs": len(state["patches"]), "model_calls": state["model_calls"],
                             "execution_seconds": round(sum(a["seconds"] for a in state["attempts"]), 3),
                             "verification": result.get("verification", {}).get("status"),
                             "events": [e["node"] for e in state["events"]],
                             "changes": [p["changes"] for p in state["patches"]]})
                print(case["id"], policy, state["status"], len(state["attempts"]), flush=True)
    summaries = []
    baseline = {r["id"]: r for r in rows if r["policy"] == args.policies[0]}
    for policy in args.policies:
        group = [r for r in rows if r["policy"] == policy]
        corrected = [r["id"] for r in group if r["success"] and not baseline[r["id"]]["success"]]
        corrupted = [r["id"] for r in group if not r["success"] and baseline[r["id"]]["success"]]
        summaries.append({"policy": policy, "successes": sum(r["success"] for r in group),
                          "total": len(group), "false_successes": sum(r["success"] and not
                           r["recoverable"] for r in group), "corrected_ids": corrected,
                          "corrupted_ids": corrupted, "net_gain_count": len(corrected)-len(corrupted),
                          "total_executions": sum(r["attempts"] for r in group)})
    result = {"scope": "10 handcrafted fault-injection cases, actual CPU subprocesses; "
              "not real-world reproduction performance or LLM evaluation",
              "dataset": "48 handcrafted training / 20 test sentences", "backend": args.backend,
              "seconds": round(time.monotonic()-started, 3), "summaries": summaries, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
