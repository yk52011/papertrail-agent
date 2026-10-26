# Tiny NLP smoke experiment

This is a handcrafted sentiment lexicon demo, NOT a paper reproduction benchmark.

## Environment
Python 3.11+, standard library only. No pretrained model, GPU, or external packages needed.

## Dataset
The included `data/test.json` contains six synthetic English sentiment examples.
Do not interpret accuracy on these examples as real-world performance.

## Run
`python evaluate.py`

Writes `results/metrics.json` with accuracy and sample-level predictions.
The experiment manifest `experiment.json` requires accuracy >= 0.8.
No checkpoint download is required. There are no published paper metrics to reproduce.
