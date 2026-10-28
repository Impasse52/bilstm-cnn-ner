# CNN-BiLSTM Named Entity Recognition

A script-based version of the academic NER project. The code trains a CNN-BiLSTM token classifier on CoNLL-2003 or CoNLL-2012, evaluates entity-level precision/recall/F1, and saves the model and experiment artifacts.

## Project structure

```text
.
├── configs/
│   └── default.json          # Hyperparameters
├── data/
│   └── conll003-englishversion/
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
├── src/
│   ├── datasets.py           # Dataset readers
│   ├── evaluation.py         # Original evaluation helpers
│   ├── model.py              # CNN-BiLSTM model definitions
│   ├── pipeline.py           # Vocabulary, features and tensor preparation
│   └── utils.py
├── train.py                  # CLI training + evaluation entry point
├── requirements.txt
└── artifacts/                # Created at runtime (models and metrics)
```

The original notebooks, paper, slides, results sheet, and included CoNLL-2003 data are retained for reference. The scripts are the recommended entry point for new runs.

## Setup

Python 3.10–3.11 is a sensible starting point for TensorFlow compatibility.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the [GloVe 6B vectors](https://nlp.stanford.edu/projects/glove/) and place `glove.6B.100d.txt` at:

```text
data/glove.6B/glove.6B.100d.txt
```

The path is configurable with `--embeddings`. The GloVe file is not included because it is large and has separate distribution terms.

## Train and evaluate CoNLL-2003

The included data folder is used by default:

```bash
python train.py --dataset conll2003
```

Training combines `train.txt` and `valid.txt`, then evaluates against `test.txt`.

## Train and evaluate CoNLL-2012

Obtain the CoNLL-2012/OntoNotes data separately (it is not included in this archive), unpack it under `data/conll-2012/`, and ensure the expected `v12/` directory contains the `.gold_conll` files. Then run:

```bash
python train.py --dataset conll2012
```

You can specify a different dataset directory:

```bash
python train.py --dataset conll2012 --data-dir /path/to/conll-2012
```

## Common options

```bash
python train.py --dataset conll2003 --epochs 20 --batch-size 64
python train.py --dataset conll2003 --config configs/default.json
python train.py --dataset conll2003 --embeddings data/glove.6B/glove.6B.100d.txt --output-dir artifacts/experiment_01 --seed 123
```

Hyperparameters live in `configs/default.json`. CLI values override the configured epoch and batch size.

## Outputs

Each run creates the following files in `artifacts/` (or the selected output directory):

- `<dataset>_cnn_bilstm.keras` — trained Keras model
- `<dataset>_metrics.json` — precision, recall, F1 and per-entity report
- `<dataset>_labels.json` — label-to-index mapping
- `<dataset>_vocabulary.json` — word, character, casing and phrase-part mappings

## Model overview

- Pretrained GloVe word embeddings (frozen)
- Token casing features
- Character-level CNN features
- POS/phrase-part features for CoNLL-2003
- Bidirectional LSTM sequence encoder
- Token-level softmax classifier

## Notes and limitations

- This is a research/academic baseline, not a production-ready service.
- The implementation pads/truncates sequences to 52 tokens and character sequences to 52 characters, reflecting the original pipeline; longer sentences may be truncated.
- CoNLL-2012 preprocessing expects the original OntoNotes-style `.gold_conll` layout.
- Results can vary with TensorFlow/Keras version, hardware, and random seed. Re-run evaluation before quoting a score in a CV.
