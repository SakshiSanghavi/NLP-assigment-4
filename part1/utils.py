import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# Typo probability used for the full Q2 and Q3 runs.
TYPO_RATE = 0.45
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS = {}
for row in ROWS:
    for i, key in enumerate(row):
        NEIGHBORS[key] = row[max(i - 1, 0):i] + row[i + 1:i + 2]


def custom_transform(example):
    """Replace one letter in some words with a neighboring QWERTY key."""
    rng = random.Random(example["text"])  # same text -> same typos
    words = example["text"].split()

    for i, word in enumerate(words):
        if len(word) >= 4 and word.isalpha() and rng.random() < TYPO_RATE:
            pos = rng.randrange(len(word))
            key = word[pos].lower()
            if key in NEIGHBORS:
                new = rng.choice(NEIGHBORS[key])
                if word[pos].isupper():
                    new = new.upper()
                words[i] = word[:pos] + new + word[pos + 1:]

    example["text"] = " ".join(words)
    return example
