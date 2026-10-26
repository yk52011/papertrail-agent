"""Small, actual ML training pipeline for the controlled recovery workload."""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

settings = json.loads(Path("config.json").read_text())
training = json.loads(Path(settings["train_path"]).read_text())
test = json.loads(Path(settings["test_path"]).read_text())
column = settings["text_column"]
train_texts = [row[column] for row in training]
labels = [row["label"] for row in training]
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, settings["ngram_max"]), min_df=settings["min_df"]),
    LogisticRegression(random_state=42, max_iter=200, class_weight="balanced"),
)
model.fit(train_texts, labels)
predicted = model.predict([row[column] for row in test]).tolist()
truth = [row["label"] for row in test]
result = {"accuracy": accuracy_score(truth, predicted),
          "macro_f1": f1_score(truth, predicted, average="macro"),
          "train_count": len(training), "test_count": len(test),
          "predictions": [{"id": row["id"], "prediction": int(value)}
                          for row, value in zip(test, predicted)]}
if settings["write_results"]:
    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "predictions"}))
