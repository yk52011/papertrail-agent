# Sentiment Lab: controlled recovery workload

A real CPU training pipeline: scikit-learn TF-IDF + LogisticRegression.
Training and test sentences are handcrafted English examples, not a published dataset.
No research novelty or external benchmark performance is implied.

## Run

Python 3.11+ and scikit-learn are required. Run `python train.py`.
Settings live in config.json. Results are written to results/metrics.json.
The repository starts with two deliberately incorrect settings for the recovery demo.

## Data configuration

The real training file is `data/train.json`; set `train_path` to `data/train.json`.
Rows use the `text` column; set `text_column` to `text`, not `sentence`.
The test split is `data/test.json`. It must not be used for training or parameter selection.

## Feature configuration

`min_df` is an integer count and must be at least 1; set `min_df` to 1 for this small corpus.
`ngram_max` must be at least 1; set `ngram_max` to 2 for unigrams and bigrams.
The seed is fixed at 42 for deterministic comparisons.

## Output contract

Set `write_results` to true to save predictions and metrics to results/metrics.json.
Each prediction must contain a unique test sample `id` and an integer `prediction` label.
The supervisor independently recomputes accuracy and macro F1 from saved predictions.
Reported values must agree with the independent calculation within 1e-6.
The required macro F1 is 0.75 on this handcrafted demonstration split.

## Recovery scope

task.json declares the only configuration keys and candidate values the supervisor may change.
The agent cannot change the test labels, thresholds, commands or source code.
Missing paths, wrong column names and invalid feature settings are injected failures;
they evaluate control flow and verification, not general autonomous debugging ability.
