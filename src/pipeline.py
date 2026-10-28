import string

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def unpack_sentences(sent_lists: list) -> tuple:
    # construct vocabulary and labels set by taking words from each partition
    labels, vocab, phrase_parts = set(), set(), set()
    for dataset in sent_lists:
        for sentence in dataset:
            for token, label, _, _, phrase_part in sentence:
                labels.add(label)
                vocab.add(token.lower())
                phrase_parts.add(phrase_part)

    return labels, vocab, phrase_parts


def unpack_sentences_conll2012(sent_lists: list) -> tuple:
    # construct vocabulary and labels set by taking words from each partition
    labels, vocab = set(), set()
    for dataset in sent_lists:
        for sentence in dataset:
            for token, label, _, _ in sentence:
                labels.add(label)
                vocab.add(token.lower())

    return labels, vocab


def build_part_of_phrase_lu(phrase_parts: list):
    # create phrase_parts lookup
    phrase_lu = {"PADDING": 0}
    for p_part in phrase_parts:
        phrase_lu[p_part] = len(phrase_lu)

    # create casing lookup
    phrase_embeddings = np.identity(len(phrase_lu), dtype="float32")

    return phrase_lu, phrase_embeddings


def build_labels_lu(labels: list) -> dict:
    # create labels lookup
    labels_lu = {"PADDING": 0}
    for label in labels:
        labels_lu[label] = len(labels_lu)

    return labels_lu


caps_lu = {
    "allCaps": 0,
    "upperInitial": 1,
    "lowercase": 2,
    "mixedCaps": 3,
}


def extract_case_features(w):
    if w.isupper():
        return caps_lu["allCaps"]
    if w.istitle():
        return caps_lu["upperInitial"]
    if w.islower():
        return caps_lu["lowercase"]

    return caps_lu["mixedCaps"]


def load_word_embeddings(data_dir: str, vocab: dict) -> tuple:
    # create words lookup and read word embeddings
    words_lu = {}
    word_emb = []

    emb_file = open(data_dir, encoding="utf-8")

    for line in emb_file:
        split = line.strip().split(" ")
        word = split[0]

        # Add padding+unknown
        if len(words_lu) == 0:
            words_lu["PADDING_TOKEN"] = len(words_lu)

            # add a zero vector for padding tokens
            vector = np.zeros(len(split) - 1)
            word_emb.append(vector)

            words_lu["UNKNOWN_TOKEN"] = len(words_lu)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            word_emb.append(vector)

        if word.lower() in vocab:
            vector = np.array([float(num) for num in split[1:]])
            word_emb.append(vector)
            words_lu[split[0]] = len(words_lu)

    word_emb = np.array(word_emb)

    return words_lu, word_emb


def create_char_lu() -> dict:
    # create character lookup
    char_lu = {"PADDING": 0, "UNKNOWN": 1}
    for c in string.printable:
        char_lu[c] = len(char_lu)

    return char_lu


def sentences_to_matrix(
    sentences: list,
    caps_lu: dict,
    words_lu: dict,
    char_lu: dict,
    phrase_lu: dict,
    labels_lu: dict,
):
    unknown_idx = words_lu["UNKNOWN_TOKEN"]

    dataset = []
    for sentence in sentences:
        word_idxs, case_idxs, char_idxs, label_idxs, phrase_idxs = [], [], [], [], []

        for word, label, char, _, phrase_part in sentence:
            # insert word idx if in vocabulary
            if word in words_lu:
                wordIdx = words_lu[word]
            elif word.lower() in words_lu:
                wordIdx = words_lu[word.lower()]
            else:
                wordIdx = unknown_idx

            # Get the label and map to int
            word_idxs.append(wordIdx)
            case_idxs.append(extract_case_features(word))

            # add out of dictionary characters to lookup table
            word_chars = []
            for x in char:
                if x not in char_lu:
                    char_lu[x] = len(char_lu)

                word_chars.append(char_lu[x])

            char_idxs.append(word_chars)
            phrase_idxs.append(phrase_lu[phrase_part])
            label_idxs.append(labels_lu[label])

        dataset.append(
            [
                np.array(word_idxs),
                np.array(case_idxs),
                pad_sequences(char_idxs, padding="post", maxlen=52),
                np.array(label_idxs),
                np.array(phrase_idxs),
            ]
        )

    return dataset


def sentences_to_matrix_conll2012(
    sentences: list,
    caps_lu: dict,
    words_lu: dict,
    char_lu: dict,
    labels_lu: dict,
):
    unknown_idx = words_lu["UNKNOWN_TOKEN"]

    dataset = []
    for sentence in sentences:
        word_idxs, case_idxs, char_idxs, label_idxs = [], [], [], []

        for word, label, char, _ in sentence:
            # insert word idx if in vocabulary
            if word in words_lu:
                wordIdx = words_lu[word]
            elif word.lower() in words_lu:
                wordIdx = words_lu[word.lower()]
            else:
                wordIdx = unknown_idx

            # Get the label and map to int
            word_idxs.append(wordIdx)
            case_idxs.append(extract_case_features(word))

            # add out of dictionary characters to lookup table
            word_chars = []
            for x in char:
                if x not in char_lu:
                    char_lu[x] = len(char_lu)

                word_chars.append(char_lu[x])

            char_idxs.append(word_chars)
            label_idxs.append(labels_lu[label])

        dataset.append(
            [
                np.array(word_idxs),
                np.array(case_idxs),
                pad_sequences(char_idxs, padding="post", maxlen=52),
                np.array(label_idxs),
            ]
        )

    return dataset


def pad_dataset(ds: list) -> tuple:
    tokens, caps, chars, labels, phrasetags = [], [], [], [], []

    for s in ds:
        tokens.append(s[0])
        caps.append(s[1])
        chars.append(s[2])
        labels.append(s[3])
        phrasetags.append(s[4])

    tokens = pad_sequences(tokens, padding="post", maxlen=52)
    caps = pad_sequences(caps, padding="post", maxlen=52)
    chars = pad_sequences(chars, padding="post", maxlen=52)
    labels = pad_sequences(labels, padding="post", maxlen=52)
    phrasetags = pad_sequences(phrasetags, padding="post", maxlen=52)

    return tokens, caps, chars, labels, phrasetags


def pad_dataset_conll2012(ds: list) -> tuple:
    tokens, caps, chars, labels = [], [], [], []

    for s in ds:
        tokens.append(s[0])
        caps.append(s[1])
        chars.append(s[2])
        labels.append(s[3])

    tokens = pad_sequences(tokens, padding="post", maxlen=52)
    caps = pad_sequences(caps, padding="post", maxlen=52)
    chars = pad_sequences(chars, padding="post", maxlen=52)
    labels = pad_sequences(labels, padding="post", maxlen=52)

    return tokens, caps, chars, labels
