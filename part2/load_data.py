from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

DATA_FOLDER = Path(__file__).resolve().parent / "data"
TOKENIZER_NAME = "google-t5/t5-small"
TASK_PREFIX = "translate English to SQL: "
PAD_IDX = 0
LABEL_PAD_IDX = -100


@lru_cache(maxsize=1)
def load_tokenizer():
    return T5TokenizerFast.from_pretrained(TOKENIZER_NAME)


def load_split(data_folder, split):
    data_folder = Path(data_folder)
    questions = load_lines(data_folder / f"{split}.nl")
    if split == "test":
        return questions, None
    queries = load_lines(data_folder / f"{split}.sql")
    if len(questions) != len(queries):
        raise ValueError(f"{split}: found {len(questions)} questions and {len(queries)} SQL queries")
    return questions, queries


class T5Dataset(Dataset):

    def __init__(self, data_folder, split, tokenizer=None):
        if tokenizer is None:
            tokenizer = load_tokenizer()
        self.encoder_ids, self.sql_ids = self.process_data(data_folder, split, tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        questions, queries = load_split(data_folder, split)
        encoder_ids = tokenizer(
            [TASK_PREFIX + question for question in questions], add_special_tokens=True
        )["input_ids"]
        sql_ids = None
        if queries is not None:
            sql_ids = tokenizer(queries, add_special_tokens=True)["input_ids"]
        return encoder_ids, sql_ids

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        encoder_ids = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        if self.sql_ids is None:
            return encoder_ids

        # T5-small uses its pad id as the decoder start token.
        sql_ids = self.sql_ids[idx]
        decoder_inputs = torch.tensor([PAD_IDX] + sql_ids[:-1], dtype=torch.long)
        decoder_targets = torch.tensor(sql_ids, dtype=torch.long)
        return encoder_ids, decoder_inputs, decoder_targets


def pad_encoder(sequences):
    encoder_ids = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX)
    return encoder_ids, encoder_mask

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_sequences, decoder_sequences, target_sequences = zip(*batch)
    encoder_ids, encoder_mask = pad_encoder(encoder_sequences)
    decoder_inputs = pad_sequence(decoder_sequences, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(target_sequences, batch_first=True, padding_value=LABEL_PAD_IDX)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, decoder_inputs[:, :1]

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_mask = pad_encoder(batch)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, tokenizer=None):
    dset = T5Dataset(DATA_FOLDER, split, tokenizer)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    tokenizer = load_tokenizer()
    train_loader = get_dataloader(batch_size, "train", tokenizer)
    dev_loader = get_dataloader(test_batch_size, "dev", tokenizer)
    test_loader = get_dataloader(test_batch_size, "test", tokenizer)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_prompting_data(data_folder):
    train_x, train_y = load_split(data_folder, "train")
    dev_x, dev_y = load_split(data_folder, "dev")
    test_x, _ = load_split(data_folder, "test")
    return train_x, train_y, dev_x, dev_y, test_x


def print_data_statistics():
    tokenizer = load_tokenizer()
    before, after = {}, {}
    for split in ("train", "dev"):
        questions, queries = load_split(DATA_FOLDER, split)
        before[split] = (
            tokenizer(questions, add_special_tokens=False)["input_ids"],
            tokenizer(queries, add_special_tokens=False)["input_ids"],
        )
        dataset = T5Dataset(DATA_FOLDER, split, tokenizer)
        after[split] = (dataset.encoder_ids, dataset.sql_ids)

    def mean_length(sequences):
        return f"{sum(map(len, sequences)) / len(sequences):.2f}"

    def vocabulary_size(sequences):
        return len({token for sequence in sequences for token in sequence})

    for title, splits in (("Before preprocessing", before), ("After preprocessing", after)):
        train_nl, train_sql = splits["train"]
        dev_nl, dev_sql = splits["dev"]
        print(f"\n{title}")
        print(f"{'Statistic':<32} {'Train':>8} {'Dev':>8}")
        rows = (
            ("Number of examples", len(train_nl), len(dev_nl)),
            ("Mean sentence length", mean_length(train_nl), mean_length(dev_nl)),
            ("Mean SQL query length", mean_length(train_sql), mean_length(dev_sql)),
            ("Vocabulary size (NL)", vocabulary_size(train_nl), vocabulary_size(dev_nl)),
            ("Vocabulary size (SQL)", vocabulary_size(train_sql), vocabulary_size(dev_sql)),
        )
        for name, train_value, dev_value in rows:
            print(f"{name:<32} {train_value:>8} {dev_value:>8}")


if __name__ == "__main__":
    print_data_statistics()
