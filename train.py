from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.datasets import load_conll2003, load_conll2012
from src.model import create_model, create_model_conll2012
from src.pipeline import (
    build_labels_lu,
    build_part_of_phrase_lu,
    caps_lu,
    create_char_lu,
    load_word_embeddings,
    pad_dataset,
    pad_dataset_conll2012,
    sentences_to_matrix,
    sentences_to_matrix_conll2012,
    unpack_sentences,
    unpack_sentences_conll2012,
)

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("conll2003", "conll2012"), default="conll2003"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Dataset directory (defaults to ./data/...)"
    )
    parser.add_argument("--embeddings", type=Path, help="Path to a GloVe text file")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "default.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--epochs", type=int, help="Override configured epoch count")
    parser.add_argument("--batch-size", type=int, help="Override configured batch size")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def predict_labels(model, dataset, label_lookup, dataset_name):
    """Predict one sentence at a time so sentence lengths are not confused with padding."""
    reverse_labels = {idx: label for label, idx in label_lookup.items()}
    y_true, y_pred = [], []
    for item in dataset:
        if dataset_name == "conll2003":
            words, casing, chars, labels, phrase_tags = item
            inputs = [
                words[None, :],
                casing[None, :],
                chars[None, :, :],
                phrase_tags[None, :],
            ]
        else:
            words, casing, chars, labels = item
            inputs = [words[None, :], casing[None, :], chars[None, :, :]]
        probabilities = model.predict(inputs, verbose=0)[0]
        predicted_ids = probabilities.argmax(axis=-1)[: len(labels)]
        y_true.append([reverse_labels[int(i)] for i in labels])
        y_pred.append([reverse_labels[int(i)] for i in predicted_ids])

    return y_true, y_pred


def main():
    args = parse_args()
    with (args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    hyper = config["model"].copy()
    if args.epochs is not None:
        hyper["EPOCHS"] = args.epochs
    if args.batch_size is not None:
        hyper["BATCH_SIZE"] = args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = args.data_dir or (
        ROOT
        / "data"
        / ("conll003-englishversion" if args.dataset == "conll2003" else "conll-2012")
    )
    embedding_path = args.embeddings or (ROOT / "data" / hyper["WORD_EMB_DIR"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Embeddings: {embedding_path}")

    phrase_lookup = {}
    if args.dataset == "conll2003":
        train_sentences, test_sentences = load_conll2003(str(data_dir))
        labels, vocab, phrase_parts = unpack_sentences([train_sentences])
        phrase_lookup, phrase_embeddings = build_part_of_phrase_lu(phrase_parts)
        labels_lookup = build_labels_lu(labels)
        case_embeddings = np.identity(len(caps_lu), dtype="float32")
        words_lookup, word_embeddings = load_word_embeddings(str(embedding_path), vocab)
        char_lookup = create_char_lu()

        train_data = sentences_to_matrix(
            train_sentences,
            caps_lu,
            words_lookup,
            char_lookup,
            phrase_lookup,
            labels_lookup,
        )
        test_data = sentences_to_matrix(
            test_sentences,
            caps_lu,
            words_lookup,
            char_lookup,
            phrase_lookup,
            labels_lookup,
        )
        model = create_model(
            hyper,
            word_embeddings,
            case_embeddings,
            phrase_embeddings,
            char_lookup,
            labels_lookup,
        )
        tokens, casing, chars, label_ids, phrase_ids = pad_dataset(train_data)
        x_train, y_train = [tokens, casing, chars, phrase_ids], label_ids
        validation_data = None
    else:
        train_sentences, dev_sentences, test_sentences = load_conll2012(str(data_dir))
        labels, vocab = unpack_sentences_conll2012([train_sentences])
        labels_lookup = build_labels_lu(labels)
        case_embeddings = np.identity(len(caps_lu), dtype="float32")
        words_lookup, word_embeddings = load_word_embeddings(str(embedding_path), vocab)
        char_lookup = create_char_lu()

        train_data = sentences_to_matrix_conll2012(
            train_sentences, caps_lu, words_lookup, char_lookup, labels_lookup
        )
        dev_data = sentences_to_matrix_conll2012(
            dev_sentences, caps_lu, words_lookup, char_lookup, labels_lookup
        )
        test_data = sentences_to_matrix_conll2012(
            test_sentences, caps_lu, words_lookup, char_lookup, labels_lookup
        )
        model = create_model_conll2012(
            hyper, word_embeddings, case_embeddings, char_lookup, labels_lookup
        )
        tokens, casing, chars, label_ids = pad_dataset_conll2012(train_data)
        dev_tokens, dev_casing, dev_chars, dev_label_ids = pad_dataset_conll2012(
            dev_data
        )
        x_train, y_train = [tokens, casing, chars], label_ids
        validation_data = ([dev_tokens, dev_casing, dev_chars], dev_label_ids)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=hyper.get("EARLY_STOPPING_PATIENCE", 5),
            restore_best_weights=True,
        ),
    ]
    model.fit(
        x_train,
        y_train,
        epochs=int(hyper["EPOCHS"]),
        batch_size=int(hyper["BATCH_SIZE"]),
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    y_true, y_pred = predict_labels(model, test_data, labels_lookup, args.dataset)
    metrics = {
        "dataset": args.dataset,
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "epochs_configured": int(hyper["EPOCHS"]),
        "batch_size": int(hyper["BATCH_SIZE"]),
        "seed": args.seed,
    }
    print("\nEvaluation")
    print(metrics["classification_report"])
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")

    model_path = args.output_dir / f"{args.dataset}_cnn_bilstm.keras"
    metrics_path = args.output_dir / f"{args.dataset}_metrics.json"
    lookup_path = args.output_dir / f"{args.dataset}_labels.json"
    model.save(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lookup_path.write_text(json.dumps(labels_lookup, indent=2), encoding="utf-8")
    (args.output_dir / f"{args.dataset}_vocabulary.json").write_text(
        json.dumps(
            {
                "words": words_lookup,
                "characters": char_lookup,
                "phrase_parts": phrase_lookup,
                "casing": caps_lu,
                "dataset": args.dataset,
                "max_char_length": 52,
                "max_sequence_length": 52,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved labels:  {lookup_path}")
    print(
        "Saved vocabulary and preprocessing data:",
        args.output_dir / (args.dataset + "_vocabulary.json"),
    )


if __name__ == "__main__":
    main()
