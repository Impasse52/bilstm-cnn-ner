import numpy as np
from keras.utils import Progbar


def predict_dataset(dataset: list, model) -> tuple:
    y_true, y_pred = [], []
    progress = Progbar(len(dataset))

    for i, data in enumerate(dataset):
        tokens, casing, chars, labels, phrasetags = data

        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        chars = np.asarray([chars])
        phrasetags = np.asarray([phrasetags])

        pred = model.predict([tokens, casing, chars, phrasetags], verbose=False)[0]
        pred = pred.argmax(axis=-1)

        y_true.append(labels)
        y_pred.append(pred)

        progress.update(i + 1)

    return y_true, y_pred


def compute_precision(y_pred: list, y_true: list) -> float:
    # ensure the length of predicted and true lists are the same
    assert len(y_pred) == len(y_true)

    correct_count = 0
    count = 0

    # iterate over each sentence in the lists
    for sentence_idx in range(len(y_pred)):
        guessed = y_pred[sentence_idx]
        correct = y_true[sentence_idx]

        # ensure the length of guessed and correct sentences are the same
        assert len(guessed) == len(correct)

        idx = 0
        while idx < len(guessed):
            # check for the beginning of a named entity in the guessed labels
            if guessed[idx][0] == "B":
                count += 1
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctly_found = True

                    # check subsequent "I" labels in the entity
                    while idx < len(guessed) and guessed[idx][0] == "I":
                        if guessed[idx] != correct[idx]:
                            correctly_found = False
                        idx += 1

                    # ensure the entity in correct labels is also complete
                    if idx < len(guessed):
                        if correct[idx][0] == "I":
                            correctly_found = False

                    # if the entire entity was correctly guessed, increment the correct count
                    if correctly_found:
                        correct_count += 1
                else:
                    idx += 1
            else:
                idx += 1

    # calculate precision if any entities were found
    precision = 0
    if count > 0:
        precision = float(correct_count) / count

    return precision


def evaluate_pred(y_pred: list, y_true: list) -> tuple:
    pre = compute_precision(y_pred, y_true)
    rec = compute_precision(y_true, y_pred)
    f1 = 0

    if (rec + pre) > 0:
        f1 = 2.0 * pre * rec / (pre + rec)

    return pre, rec, f1
