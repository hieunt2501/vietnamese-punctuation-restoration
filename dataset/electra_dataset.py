import re

import torch
from torch.nn.utils.rnn import pad_sequence


class Collator:
    def __init__(self, tokenizer, max_len=512, label_strategy="first"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_strategy = label_strategy
        self.label_dict = {",": 1,
                           ".": 2,
                           "!": 3,
                           "?": 4,
                           ":": 5,
                           ";": 6}

    def _get_token_and_label(self, sample):
        sample_tokens, sample_labels = list(), list()
        text = sample["text"].lower()
        text = re.sub(r"\d+([./,]\d+)?", "<NUM>", text)
        org_tokens = text.split()

        # create eligible sample
        for token in org_tokens:
            label = self.label_dict.get(token, 0)
            if label and sample_labels:
                sample_labels[-1] = label
            elif not label:
                sample_labels.append(label)
                sample_tokens.append(token)



        return sample_tokens, sample_labels

    def _create_sample(self, sample):
        valid_ids = list()
        sample_tokens, sample_labels = self._get_token_and_label(sample)
        assert len(sample_tokens) == len(sample_labels)
        sample_labels = [0] + sample_labels[:self.max_len-2] + [0]

        for token in sample_tokens:
            tokenized_token = self.tokenizer.tokenize(token)
            # for idx in range(len(tokenized_token)):
            if self.label_strategy == "first":
                valid_ids.extend([1] + [0]*(len(tokenized_token) - 1))
            else:
                valid_ids.extend([0]*(len(tokenized_token) - 1) + [1])
        valid_ids = [1] + valid_ids[:self.max_len - 2] + [1]
        subword_padding = (len(valid_ids) - len(sample_labels))
        label_masks = [1]*len(sample_labels) + [0] * subword_padding
        sample_labels += [0] * subword_padding
        return sample_tokens, sample_labels, label_masks, valid_ids

    def __call__(self, batch):
        batch_tokens, batch_labels, batch_label_masks, batch_valid_ids = list(), list(), list(), list()
        for sample in batch:
            tokens, labels, label_masks, valid_ids = self._create_sample(sample)
            batch_tokens.append(tokens)
            batch_labels.append(torch.tensor(labels))
            batch_label_masks.append(torch.tensor(label_masks))
            batch_valid_ids.append(torch.tensor(valid_ids))

        sentence_lengths = [len(valid_ids) for valid_ids in batch_valid_ids]
        batch_valid_ids = pad_sequence(batch_valid_ids, batch_first=True, padding_value=0)

        # padding to match ids length
        batch_label_masks = pad_sequence(batch_label_masks, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0)


        output = self.tokenizer(batch_tokens,
                                is_split_into_words=True,
                                max_length=self.max_len,
                                truncation=True,
                                return_tensors="pt",
                                padding="longest")
        input_ids, attention_mask = output["input_ids"], output["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels,
            "valid_ids": batch_valid_ids,
            "label_mask": batch_label_masks,
            "sentence_lengths": sentence_lengths
        }


class ElectraCollator(Collator):
    def __init__(self, tokenizer, max_len=512, label_strategy="first"):
        super().__init__(tokenizer, max_len, label_strategy)
        classes = ["O", "COMMA", "PERIOD", "EXCLAM", "QUESTION", "COLON", "SEMICOLON"]
        self.label_dict = {class_: i for i, class_ in enumerate(classes)}

    def _get_token_and_label(self, sample):
        sample_tokens, sample_labels = sample["text"].lower().split(), sample["label"].split()
        sample_labels = [self.label_dict[label] for label in sample_labels]

        return sample_tokens, sample_labels
