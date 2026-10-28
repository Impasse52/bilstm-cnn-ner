# CNN-BiLSTM Named Entity Recognition

A reworked script-based version of a former NER project realized in the context of an university exam. This script trains a CNN-BiLSTM token classifier on CoNLL-2003 or CoNLL-2012, evaluates entity-level precision/recall/F1, and saves the model and experiment artifacts. The original version can be found in `notebooks`.

This project is based on [Named Entity Recognition with Bidirectional LSTM-CNNs (Chiu and Nichols, 2016)](https://arxiv.org/abs/1511.08308).

## Setup


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the [GloVe 6B vectors](https://nlp.stanford.edu/projects/glove/) and place `glove.6B.100d.txt` at:

```text
data/glove.6B/glove.6B.100d.txt
```

The path is configurable with `--embeddings`. The GloVe file is not included because it is large and has separate distribution terms.

The included data folder is used by default:

```bash
python train.py --dataset conll2003
```


## Results

The CNN-BiLSTM model was evaluated on the CoNLL-2003 and CoNLL-2012 datasets. Experiments explored different optimizers, GloVe embedding dimensions, LSTM units, dropout rates, and convolution widths.

### Best recorded results

| Dataset | Optimizer | Embeddings | Precision | Recall | F1-score |
|---|---|---|---:|---:|---:|
| CoNLL-2003 | Adam | GloVe 6B 100d | **89.4%** | **90.2%** | **89.8%** |
| CoNLL-2012 | RMSprop | GloVe 6B 200d | **70.4%** | **69.5%** | **69.9%** |

### Experimental results

| Dataset | Optimizer | Embeddings | Epochs | Batch size | POS features | LSTM units | Dropout | Conv width | Precision | Recall | F1-score |
|---|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| CoNLL-2003 | Adam | GloVe 100d | 80 | 128 | Yes | 300 | 0.68 | 3 | 89.4% | 90.2% | **89.8%** |
| CoNLL-2003 | RMSprop | GloVe 100d | 80 | 9 | No | 275 | 0.68 | 3 | 89.1% | 89.7% | 89.4% |
| CoNLL-2003 | RMSprop | GloVe 100d | 80 | 128 | No | 275 | 0.68 | 3 | 89.1% | 90.2% | 89.7% |
| CoNLL-2003 | RMSprop | GloVe 50d | 80 | 128 | No | 275 | 0.68 | 3 | 87.8% | 89.5% | 88.6% |
| CoNLL-2003 | RMSprop | GloVe 200d | 80 | 128 | No | 275 | 0.68 | 3 | 88.5% | 89.9% | 89.2% |
| CoNLL-2003 | RMSprop | GloVe 300d | 80 | 128 | No | 275 | 0.68 | 3 | 87.9% | 89.8% | 88.9% |
| CoNLL-2003 | RMSprop | GloVe 200d | 80 | 128 | Yes | 275 | 0.68 | 3 | 88.5% | 89.6% | 89.0% |
| CoNLL-2003 | Nadam | GloVe 100d | 80 | 128 | Yes | 275 | 0.68 | 3 | 88.3% | 89.9% | 89.1% |
| CoNLL-2003 | SGD | GloVe 50d | 80 | 9 | No | 275 | 0.68 | 3 | 81.5% | 79.9% | 80.7% |
| CoNLL-2003 | Adam | GloVe 100d | 80 | 128 | Yes | 275 | 0.68 | 3 | 89.1% | 89.9% | 89.5% |
| CoNLL-2003 | Adam | GloVe 100d | 80 | 128 | Yes | 325 | 0.68 | 3 | 89.1% | 90.2% | 89.6% |
| CoNLL-2003 | Adam | GloVe 100d | 80 | 128 | Yes | 300 | 0.68 | 4 | 88.4% | 89.7% | 89.0% |
| CoNLL-2003 | Adam | GloVe 100d | 80 | 128 | Yes | 300 | 0.50 | 4 | 89.4% | 90.1% | 89.7% |
| CoNLL-2012 | RMSprop | GloVe 200d | 18 | 9 | No | 300 | 0.68 | 3 | 70.4% | 69.5% | 69.9% |
| CoNLL-2012 | RMSprop | GloVe 100d | 18 | 9 | No | 300 | 0.68 | 3 | 70.0% | 56.0% | 62.0% |

