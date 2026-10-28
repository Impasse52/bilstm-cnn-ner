import glob
import re

try:
    from .utils import flatten_list
except ImportError:  # Support running modules directly from src/
    from utils import flatten_list


def convert_to_bio(sentence: list) -> list:
    """Utility function that converts OntoNotes5.0/CoNLL2012's tagging to BIO"""

    bio_sentence = []
    i, inside = 0, False

    while i < len(sentence):
        if "*" not in sentence[i][1] or "(" in sentence[i][1]:
            tag = sentence[i][1].replace("(", "").replace(")", "").replace("*", "")

        # match whole tags
        if sentence[i][1].startswith("(") and sentence[i][1].endswith(")"):
            # print("B", sentence[i])
            bio_sentence.append((sentence[i][0], "B-" + tag))

        # match beginning tags
        elif sentence[i][1].startswith("(") and not sentence[i][1].endswith(")"):
            inside = True
            # print("B", sentence[i])
            bio_sentence.append((sentence[i][0], "B-" + tag))

        elif sentence[i][1].endswith(")"):
            # print("I", sentence[i])
            bio_sentence.append((sentence[i][0], "I-" + tag))

            inside = False
        else:
            if inside:
                # print("I", sentence[i])
                bio_sentence.append((sentence[i][0], "I-" + tag))

            else:
                # print("O", sentence[i])
                bio_sentence.append((sentence[i][0], "O"))

        # print()
        i += 1

    return bio_sentence


def build_ner_dataset_2012(filepath: str) -> list:
    with open(filepath) as f:
        lines = f.readlines()

    sentences, curr_sentence = [], []
    for line in lines:
        if line.startswith("#begin") or line.startswith("#end"):
            continue

        # sentence end
        if line == "\n":
            if curr_sentence != []:
                # convert sentence to BIO and append it to output list
                bio_sentence = convert_to_bio(curr_sentence)
                curr_sentence = [
                    (
                        curr_sentence[i][0],
                        bio_sentence[i][1],
                        curr_sentence[i][2],
                        curr_sentence[i][3],
                    )
                    for i in range(len(bio_sentence))
                ]
                sentences.append(curr_sentence)

            curr_sentence = []
            continue

        line = re.sub(r"  *", " ", line).split(" ")

        token, label = line[3], line[10]

        curr_sentence.append((token, label, list(token), token))

    return sentences


def load_conll2012(data_dir: str) -> tuple:
    files = glob.glob(f"{data_dir}/v12/**", recursive=True)
    files = list(filter(lambda x: ".gold_conll" in x, files))

    train_files = list(filter(lambda x: "train" in x, files))
    dev_files = list(filter(lambda x: "dev" in x, files))
    test_files = list(
        filter(lambda x: "test" in x and "conll-2012-test" not in x, files)
    )

    train_sentences = flatten_list(
        [build_ner_dataset_2012(train_file) for train_file in train_files]
    )
    dev_sentences = flatten_list(
        [build_ner_dataset_2012(dev_file) for dev_file in dev_files]
    )
    test_sentences = flatten_list(
        [build_ner_dataset_2012(test_file) for test_file in test_files]
    )

    return train_sentences, dev_sentences, test_sentences


def build_ner_dataset_2003(filepath: str) -> list:
    with open(filepath) as f:
        lines = f.readlines()

    sentences, curr_sentence = [], []
    for line in lines:
        if line.startswith("-DOCSTART-"):
            continue

        if line.strip() == "":
            if curr_sentence:
                sentences.append(curr_sentence)

            curr_sentence = []
            continue

        parts = line.split()
        if len(parts) < 4:
            continue
        token, _, phrase_part, label = parts[:4]
        curr_sentence.append((token, label.strip(), list(token), (token), phrase_part))

    return sentences


def load_conll2003(data_dir: str) -> tuple:
    train_sentences = build_ner_dataset_2003(
        f"{data_dir}/train.txt"
    ) + build_ner_dataset_2003(f"{data_dir}/valid.txt")
    test_sentences = build_ner_dataset_2003(f"{data_dir}/test.txt")

    return train_sentences, test_sentences
