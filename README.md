# POS Tagger Project

A local-first Chinese/English POS tagging system built around a structured averaged perceptron, Viterbi decoding, a lightweight local HTTP server, and a local web UI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.main
```

Open <http://127.0.0.1:8000> and train the demo model from the homepage.

## Features

- Convert PKU/MSR, WSJ `word/TAG`, and PTB-style corpora into CoNLL.
- Load and split corpora locally with 8:1:1 train/valid/test partitions.
- Train a structured averaged perceptron with contextual, morphological, and transition features.
- Decode with Viterbi and compare with a most-frequent-tag baseline.
- Expose `/train`, `/predict`, and `/stats` APIs plus a browser UI.
