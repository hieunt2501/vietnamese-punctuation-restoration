from ast import literal_eval
from collections import defaultdict
from typing import Union, Dict, Any

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PuncResDataset(Dataset):
    def __init__(self, tokenizer, dataset: Union[Dict[str, Any], pd.DataFrame], max_len: int = 256):
        self.texts = dataset["text"]
        self.labels = dataset["label"]
        self.tokenizer = tokenizer
        self.max_len = max_len

        if isinstance(dataset, pd.DataFrame):
            self.texts = [literal_eval(text) for text in self.texts.tolist()]
            self.labels = [literal_eval(label) for label in self.labels.tolist()]

    def mask_subword_label(self, words):
        pass

    def get_subword_2_word_idx(self, words):
        subword2word_idx = list()
        word_idx = 1
        for word in words:
            token_ids = self.tokenizer.encode([word], is_split_into_words=True, add_special_tokens=False)
            if any(token_id in self.tokenizer.all_special_ids for token_id in token_ids):
                continue
            subword2word_idx.append(word_idx)
            word_idx += len(token_ids)

        return subword2word_idx

    def __getitem__(self, item):
        subword2word_idx = self.get_subword_2_word_idx(self.texts[item])
        output = self.tokenizer(" ".join(self.texts[item]), truncation="longest_first", max_length=self.max_len)
        input_ids, attention_mask = output["input_ids"], output["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": self.labels[item],
            "subword2word_idx": subword2word_idx
        }

    def __len__(self):
        return len(self.texts)


class TokenizeCollator:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_dict = {",": 1,
                           ".": 2,
                           "!": 3,
                           "?": 4,
                           ":": 5,
                           ";": 6}

    def __call__(self, batch):
        batch = [sample["text"] for sample in batch]
        batch_tokens = [sample.split() for sample in batch]
        labels = [[self.label_dict.get(token, 0) for token in sentence] for sentence in batch]

        final_labels = list()
        for sentence, sentence_label in zip(batch_tokens, labels):
            tmp_label = [0]
            for token, label in zip(sentence, sentence_label):
                tokenized_token = self.tokenizer.tokenize(token)
                tmp_label.extend([label]*len(tokenized_token))
            tmp_label.append(0)
            final_labels.append(torch.tensor(tmp_label[:self.max_len]))

        labels = pad_sequence(final_labels, batch_first=True, padding_value=0)
        output = self.tokenizer(batch,
                                return_length=True,
                                max_length=self.max_len,
                                truncation=True,
                                return_tensors="pt",
                                padding="longest")
        input_ids, attention_mask, sentence_lengths = output["input_ids"], output["attention_mask"], output["length"]

        assert len(input_ids[0]) == len(labels[0]), f"{len(input_ids[0]), len(labels[0])}"
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sentence_lengths": sentence_lengths
        }


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_input = defaultdict(list)

        for item in batch:
            batch_input["attention_mask"].append(torch.tensor(item["attention_mask"]))
            batch_input["source_input_ids"].append((torch.tensor(item["input_ids"]), item["sentence_length"]))
            batch_input["labels"].append(torch.tensor(item["labels"]).long())

        input_ids, input_lengths = zip(*batch_input["source_input_ids"])

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(batch_input["attention_mask"], batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(batch_input["labels"], batch_first=True, padding_value=0)
        input_lengths = torch.tensor(input_lengths)

        input_lengths, sorted_idx = input_lengths.sort(0, descending=True)

        input_ids = input_ids[sorted_idx]
        attention_mask = attention_mask[sorted_idx]
        labels = labels[sorted_idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sentence_lengths": input_lengths
        }