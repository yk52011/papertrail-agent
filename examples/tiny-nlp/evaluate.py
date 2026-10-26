"""Deterministic smoke experiment, not a trained model."""

import json
import re
from pathlib import Path

positive = {"good", "wonderful", "love", "great", "excellent"}
negative = {"bad", "boring", "hate", "terrible", "awful"}
rows = json.loads(Path("data/test.json").read_text())
predictions = []
for row in rows:
    words = set(re.findall(r"[a-z]+", row["text"].lower()))
    prediction = int(len(words & positive) > len(words & negative))
    predictions.append({"id": row["id"], "prediction": prediction,
                        "label": row["label"], "correct": prediction == row["label"]})
metrics = {"accuracy": sum(p["correct"] for p in predictions) / len(predictions),
           "count": len(rows), "predictions": predictions}
Path("results").mkdir(exist_ok=True)
Path("results/metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics))
