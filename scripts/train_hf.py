import os
import logging
import pickle
from ast import literal_eval

import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    Trainer,
    AdamW,
    get_cosine_with_hard_restarts_schedule_with_warmup
)


from trainer.argument_class import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from modules.punc_restore_hf import PuncRestore
from dataset.dataset import Collator


logger = logging.getLogger(__name__)


def load_class_weight(weight_filepath):
    with open(weight_filepath, "rb") as f:
        class_weight = pickle.load(f)
    return class_weight


def calculate_class_weight(labels, ignore_index=100):
    classes, counts = np.unique(labels, return_counts=True)
    ignore_idx = np.where(classes!=ignore_index)[0]
    if ignore_index:
        classes = classes[ignore_idx]
        counts = counts[ignore_idx]
    class_weights = counts.sum() / counts / len(classes)
    class_weights = torch.from_numpy(class_weights).float()
    return class_weights


def get_dataset(data_args, cal_class_weight=False, weight_filepath=None):
    # train_dataset = pd.read_csv(data_args.train_data_file, nrows=100)
    # eval_dataset = pd.read_csv(data_args.train_data_file, nrows=10)
    # train_dataset = datasets.Dataset.from_pandas(train_dataset, split="train")
    # eval_dataset = datasets.Dataset.from_pandas(eval_dataset, split="test")

    # train_dataset = datasets.load_dataset("csv",
    #                                       data_files={"train": data_args.train_data_file},
    #                                       streaming=True)["train"]
    # eval_dataset = datasets.load_dataset("csv",
    #                                       data_files={"test": data_args.eval_data_file})["test"]
    if data_args.eval_data_file:
         data_files={"train": data_args.train_data_file,
                     "test": data_args.eval_data_file}
    else:
        data_files = data_args.train_data_file
    # data_files = {"train": data_args.train_data_file}
    dataset = datasets.load_dataset("csv",
                                    data_files=data_files,
                                    num_proc=2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    def lit_eval(examples):
        examples["labels"] = [literal_eval(example) for example in examples["labels"]]
        examples["input_ids"] = [literal_eval(example) for example in examples["input_ids"]]
        examples["attention_mask"] = [literal_eval(example) for example in examples["attention_mask"]]

        return examples

    logger.info("Preprocessing data...")
    train_dataset = train_dataset.map(lit_eval, batched=True)
    eval_dataset = eval_dataset.map(lit_eval, batched=True)

    # train_dataset.shuffle(buffer_size=1024, seed=42)
    # eval_dataset.shuffle(buffer_size=1024, seed=42)

    class_weight = None
    if weight_filepath:
        class_weight = load_class_weight(weight_filepath)
    elif cal_class_weight:
        logger.info("Calculating class weights")
        labels = train_dataset["labels"]
        labels = [_ for label in labels for _ in label]
        class_weight = calculate_class_weight(labels)
    print(f"Class weights: {class_weight}")
    return train_dataset, eval_dataset, class_weight


def get_optimizer(model, args, total_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      correct_bias=True,
                      eps=args.adam_epsilon,
                      betas=(args.adam_beta1, args.adam_beta2))

    num_warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=num_warmup_steps,
                                                                   num_training_steps=total_steps,
                                                                   num_cycles=args.num_cycles)
    return optimizer, scheduler


def main():
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exist and is not empty. Use"
            " --overwrite_output_dir to overcome"
        )

    # Set seed
    set_seed(training_args.seed)

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "Please provide a pretrained tokenizer name"
        )

    if not data_args.eval_data_file and not data_args.train_data_file:
        raise ValueError("Please provide train and eval data file")
    else:
        train_dataset, eval_dataset, class_weights = get_dataset(data_args, training_args.weighted_loss, training_args.weight_filepath)

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, return_dict=True, output_hidden_states=True)
    else:
        config = AutoConfig.from_pretrained('vinai/phobert-base', return_dict=True, output_hidden_states=True)
    model = PuncRestore(config,
                        d_model=training_args.d_model,
                        bidirectional=training_args.bidirectional,
                        n_layers=training_args.n_layers,
                        h_dim=training_args.h_dim,
                        n_class=training_args.n_class,
                        dropout=training_args.dropout,
                        class_weights=class_weights)
    if training_args.resume_from_checkpoint:
        logger.info("Loading checkpoint")
        state_dict = torch.load(training_args.resume_from_checkpoint + "/pytorch_model.bin")
        model.load_state_dict(state_dict)

    if training_args.num_freeze_layers:
        logger.info(f"Freezing {training_args.num_freeze_layers} bert layers")
        model.freeze_transformers(training_args.num_freeze_layers)

    collator = Collator(tokenizer.pad_token_id)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        # 1: COMMA, 2:  PERIOD
        precision = precision_score(labels, preds, average="macro", labels=[0, 1])
        recall = recall_score(labels, preds, average="macro", labels=[0, 1])
        f1 = f1_score(labels, preds, average="macro", labels=[0, 1])
        print(classification_report(labels, preds, zero_division=0, labels=[0, 1]))

        return {
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    # print(trainer.evaluate())
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
