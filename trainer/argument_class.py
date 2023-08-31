from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_nrows: Optional[int] = field(
        default=None, metadata={"help": "Number of training sample rows"}
    )
    eval_nrows: Optional[int] = field(
        default=None, metadata={"help": "Number of evaluation sample rows"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(
        default=1, metadata={"help": "Softmax temperature"}
    )
    use_device: Optional[str] = field(
        default="cpu", metadata={"help": "Device for model"}
    )
    num_cycles: Optional[int] = field(
        default=1, metadata={"help": "Number of cycles for hard reset scheduler"}
    )
    d_model: Optional[int] = field(
        default=1024, metadata={"help": "Dimension of linear layer"}
    )
    h_dim: Optional[int] = field(
        default=512, metadata={"help": "Hidden dimension of LSTM"}
    )
    n_layers: Optional[int] = field(
        default=2, metadata={"help": "Number of LSTM layers"}
    )
    n_class: Optional[int] = field(
        default=2, metadata={"help": "Number of classification classes"}
    )
    dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Drop out value"}
    )
    weighted_loss: Optional[bool] = field(
        default=False, metadata={"help": "Enable weighted loss"}
    )
    bidirectional: Optional[bool] = field(
        default=True, metadata={"help": "Bidirectional LSTM"}
    )
    bucket_batch_sampler: Optional[bool] = field(
        default=False, metadata={"help": "Enable bucket batch sampler"}
    )
    sorting_keys: Optional[str] = field(
        default="input_ids", metadata={"help": "Sorting keys for bucket batch sampler"}
    )
    padding_noise: Optional[float] = field(
        default=0.02, metadata={"help": "Padding noise for bucket batch sampler"}
    )
    dataloader_shuffle: Optional[bool] = field(
        default=True, metadata={"help": "Enable for dataloader"}
    )
    num_freeze_layers: Optional[int] = field(
        default=0, metadata={"help": "Number of freezed encoder layers"}
    )
    weight_filepath: Optional[str] = field(
        default="", metadata={"help": "Pre-compute class weight for loss functions"}
    )
