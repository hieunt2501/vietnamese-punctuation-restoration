import os
import logging
import pickle

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
from modules.electra import PuncElectraCrfModel
from dataset.electra_dataset import ElectraCollator, Collator


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
    train_dataset = pd.read_csv(data_args.train_data_file, nrows=data_args.train_nrows)
    eval_dataset = pd.read_csv(data_args.eval_data_file, nrows=data_args.eval_nrows)
    train_dataset = datasets.Dataset.from_pandas(train_dataset, split="train")
    eval_dataset = datasets.Dataset.from_pandas(eval_dataset, split="test")

    # if data_args.eval_data_file:
    #      data_files={"train": data_args.train_data_file,
    #                  "test": data_args.eval_data_file}
    # else:
    #     data_files = data_args.train_data_file
    # # data_files = {"train": data_args.train_data_file}
    # dataset = datasets.load_dataset("csv",
    #                                 data_files=data_files)
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["test"]

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

    special_tokens = ['<NUM>']
    tokenizer.add_tokens(special_tokens)

    if not data_args.eval_data_file and not data_args.train_data_file:
        raise ValueError("Please provide train and eval data file")
    else:
        train_dataset, eval_dataset, class_weights = get_dataset(data_args, training_args.weighted_loss, training_args.weight_filepath)

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, return_dict=True, output_hidden_states=True)
    else:
        config = AutoConfig.from_pretrained('vinai/phobert-base', return_dict=True, output_hidden_states=True)

    config.return_dict = True
    config.num_labels = training_args.n_class
    model = PuncElectraCrfModel.from_pretrained(model_args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    if training_args.resume_from_checkpoint:
        logger.info("Loading checkpoint")
        state_dict = torch.load(training_args.resume_from_checkpoint + "/pytorch_model.bin")
        model.load_state_dict(state_dict)

    collator = Collator(tokenizer)

    def preprocess_logits_for_metrics(logits, labels):
        # print(logits[0])
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(labels, tuple):
            labels = labels[0]

        if isinstance(labels, list):
            labels = np.array([l for label in labels for l in label])
        else:
            labels = labels.reshape(-1)

        if isinstance(preds, list):
            preds = np.array([p for pred in preds for p in pred])
        else:
            preds = preds.reshape(-1)

        class_labels = [0, 1, 2, 3, 4, 5, 6]
        precision = precision_score(labels, preds, average="macro", labels=class_labels)
        recall = recall_score(labels, preds, average="macro", labels=class_labels)
        f1 = f1_score(labels, preds, average="macro", labels=class_labels)
        print(classification_report(labels, preds, zero_division=0, labels=class_labels))

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
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # print(trainer.evaluate())
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
