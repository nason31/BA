import csv
import os
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional, Sequence, Union

import numpy as np
import unidecode
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups


def _load_csv_filepath(csv_filepath: str) -> list:
    """
    Loads three elements from a csv file and appends them to a list.

    Arguments:
        csv_filepath (str): Filepath to .csv file.

    Returns:
        list: 2-dimensional list containing three elements.
    """

    data = []
    with open(csv_filepath, "r") as file:
        reader = csv.reader(file, delimiter=",", quotechar='"')
        for row in reader:
            data.append([row[0], row[1], row[2]])
    return data


def read_fn_label(filename: str) -> dict:
    """
    Reads a csv file and returns a dictionary containing
    title+description: label pairs.

    Arguments:
        filename (str): Filepath to a csv file containing label, title, description.

    Returns:
        dict: {title. description: label} pairings.
    """

    text2label = {}
    data = _load_csv_filepath(filename)
    for row in data:
        label, title, desc = row[0], row[1], row[2]
        text = ". ".join([title, desc])
        text2label[text] = label

    return text2label


def read_label(filename: str) -> list:
    """
    Reads the first item from the `filename` csv filepath in each row.

    Arguments:
        filename (str): Filepath to a csv file containing label, title, description.

    Returns:
        list: Labels from the `fn` filepath.
    """

    labels = [row[0] for row in _load_csv_filepath(filename)]
    return labels


def read_fn_compress(filename: str) -> list:
    """
    Opens a compressed file and returns the contents
    and delimits the contents on new lines.

    Arguments:
        filename (str): Filepath to a compressed file.

    Returns:
        list: Compressed file contents line separated.
    """

    text = unidecode.unidecode(open(filename).read())
    text_list = text.strip().split("\n")
    return text_list


def read_torch_text_labels(dataset: list, indices: Sequence[int]) -> tuple:
    """
    Extracts the text and labels lists from a pytorch
    `dataset` on `indices`.

    Arguments:
        dataset (list): List of lists containing text and labels.
        indices (list): List of list indices to extract text and
                         labels on from `dataset`.

    Returns:
        (list, list): Text and Label pairs from `dataset` on `indices`.

    """
    text_list = []
    label_list = []

    for index in indices:
        try:
            row = dataset[index]
        except IndexError:
            row = None
            pass

        if row:
            label_list.append(row[0])
            text_list.append(row[1])

    return text_list, label_list


def load_20news() -> tuple:
    """
    Loads the 20NewsGroups dataset from `torchtext`.

    Returns:
        tuple: Tuple of Lists, with training data at index 0 and test at
               index 1.

    """

    def process(dataset):
        pairs = []
        for i in range(len(dataset.data)):
            text = dataset.data[i]
            label = dataset.target[i]
            pairs.append((label, text))
        return pairs

    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_test = fetch_20newsgroups(subset="test")
    train_ds, test_ds = process(newsgroups_train), process(newsgroups_test)
    return train_ds, test_ds

def load_r8() -> tuple:
    """
    Load the R8 dataset from huggingface datasets.
    """

    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["label"]
            text = pair["text"]
            pairs.append((label, text))
        return pairs

    ds = load_dataset("dxgp/r8")
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds

def load_r52() -> tuple:
    """
    Load the R52 dataset from huggingface datasets.
    """

    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["label"]
            text = pair["text"]
            pairs.append((label, text))
        return pairs

    ds = load_dataset("dxgp/r52")
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds

def load_kinnews_kirnews(
    dataset_name: str = "kinnews_kirnews", data_split: str = "kinnews_cleaned"
) -> tuple:
    """
    Loads the KINNEWS and KIRNEWS datasets.

    :ref: https://huggingface.co/datasets/kinnews_kirnews

    Arguments:
        dataset_name (str): Name of the dataset to be loaded.
        data_split (str): The data split to be loaded.

    Returns:
        tuple: Tuple of lists containing the training and testing datasets respectively.
    """

    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["label"]
            title = pair["title"]
            content = pair["content"]
            pairs.append((label, title + " " + content))
        return pairs

    ds = load_dataset(dataset_name, data_split)
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds


def load_swahili() -> tuple:
    """
    Loads the Swahili dataset

    Returns:
        tuple: Tuple of lists containing the training and testing datasets respectively.
    """

    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["label"]
            text = pair["text"]
            pairs.append((label, text))
        return pairs

    ds = load_dataset("swahili_news")
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds



def load_filipino() -> tuple:
    """
    Loads the Filipino dataset from huggingface datasets
    """
    
    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["dengue"]
            text = pair["text"]
            pairs.append((label, text))
        return pairs

    ds = load_dataset("dengue_filipino")
    print(ds)
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds

def load_yahooAnswers() -> tuple:
    """
    Loads the YahooAnswers dataset from huggingface datasets

    :ref:
    """
    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["topic"]
            text = pair["best_answer"]
            pairs.append((label, text))
        return pairs
    
    ds = load_dataset("yahoo_answers_topics")
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds

def load_ag_news() -> tuple:
    """
    Loads the AG News dataset from huggingface datasets

    """
    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["label"]
            text = pair["text"]
            pairs.append((label, text))
        return pairs
    
    ds = load_dataset("ag_news")
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds

def load_trec() -> tuple:
    """
    Loads the TREC dataset from huggingface datasets

    """
    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["coarse_label"]
            text = pair["text"]
            pairs.append((label, text))
        return pairs
    
    ds = load_dataset("trec")
    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds

def load_emotion() -> tuple:
    """
    Loads the Emotion dataset from huggingface datasets

    """
    def process(dataset: Iterable) -> list:
        pairs = []
        for pair in dataset:
            label = pair["label"]
            text = pair["text"]
            pairs.append((label, text))
        return pairs

    ds = load_dataset("dair-ai/emotion", download_mode="force_redownload", ignore_verifications=True)

    train_ds, test_ds = process(ds["train"]), process(ds["test"])
    return train_ds, test_ds


def pick_n_sample_from_each_class(
    filename: str, n_samples: int, idx_only: bool = False
) -> Union[list, tuple]:
    """
    Grabs a random sample of size `n_samples` for each label from the csv file
    at `filename`.

    Arguments:
        filename (str): Relative path to the file you want to load.
        n_samples (int): Number of samples to load and return for each label.
        idx_only (bool): True if you only want to return the indices of the rows
                         to load.

    Returns:
        list | tuple: List if idx_only, else tuple of samples and labels.

    """

    label2text = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []

    data = _load_csv_filepath(filename)
    for i, (label, title, description) in enumerate(data):
        text = ". ".join([title, description])
        label2text[label].append(text)
        label2idx[label].append(i)

    for class_ in label2text:
        class2count[class_] = len(label2text[class_])

    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n_samples, replace=False)
        select_text = np.array(label2text[c])[select_idx]
        select_text_idx = np.array(label2idx[c])[select_idx]
        recorded_idx += list(select_text_idx)
        result += list(select_text)
        labels += [c] * n_samples

    if idx_only:
        return recorded_idx

    return result, labels


def pick_n_sample_from_each_class_given_dataset(
    dataset: Iterable,
    n_samples: int,
    output_filename: Optional[str] = None,
    index_only: bool = False,
) -> tuple:
    """
    Grabs a random sample of size `n_samples` for each label from the dataset
    `dataset`.

    Arguments:
        dataset (Iterable): Labeled data, in ``label, text`` pairs.
        n_samples (int): Number of samples to load and return for each label.
        output_filename (str): [Optional] Where to save the recorded indices.
        index_only (bool): True if you only want to return the indices of the rows
                           to load.

    Returns:
        list | tuple: List if idx_only, else tuple of samples and labels.
    """

    label2text = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []

    for i, (label, text) in enumerate(dataset):
        label2text[label].append(text)
        label2idx[label].append(i)

    for cl in label2text:
        class2count[cl] = len(label2text[cl])

    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n_samples, replace=False)
        select_text = np.array(label2text[c])[select_idx]
        select_text_idx = np.array(label2idx[c])[select_idx]
        recorded_idx += list(select_text_idx)
        result += list(select_text)
        labels += [c] * n_samples

    if output_filename is not None:
        np.save(output_filename, np.array(recorded_idx))

    if index_only:
        return np.array(recorded_idx), labels
    print(result)
    return result, labels


def pick_n_sample_from_each_class_img(
    dataset: list, n_samples: int, flatten: bool = False
) -> tuple:
    """
    Grabs a random sample of size `n_samples` for each label from the dataset
    `dataset`.

    Arguments:
        dataset (list): Relative path to the file you want to load.
        n_samples (int): Number of samples to load and return for each label.
        flatten (bool): True if you want to flatten the images.

    Returns:
        tuple: Tuple of samples, labels, and the recorded indices.
    """

    label2img = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []  # for replication
    for i, pair in enumerate(dataset):
        img, label = pair
        if flatten:
            img = np.array(img).flatten()
        label2img[label].append(img)
        label2idx[label].append(i)

    for cl in label2img:
        class2count[cl] = len(label2img[cl])

    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n_samples, replace=False)
        select_img = np.array(label2img[c])[select_idx]
        select_img_idx = np.array(label2idx[c])[select_idx]
        recorded_idx += list(select_img_idx)
        result += list(select_img)
        labels += [c] * n_samples
    return result, labels, recorded_idx


def load_custom_dataset(directory: str, delimiter: str = "\t") -> tuple:
    def process(filename: str) -> list:
        pairs = []
        text_list = open(filename).read().strip().split("\n")
        for t in text_list:
            label, text = t.split(delimiter)
            pairs.append((label, text))
        return pairs

    test_fn = os.path.join(directory, "test.txt")
    train_fn = os.path.join(directory, "train.txt")
    train_ds, test_ds = process(train_fn), process(test_fn)
    return train_ds, test_ds
