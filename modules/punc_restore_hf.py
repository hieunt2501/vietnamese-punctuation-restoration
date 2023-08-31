from abc import ABC
from typing import Optional, Union, Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers import RobertaPreTrainedModel, RobertaModel
# from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from modules.projection import ProjectionMLP


class PuncRestore(RobertaPreTrainedModel, ABC):
    def __init__(self,
                 config,
                 d_model=1024,
                 bidirectional=True,
                 n_layers=2,
                 h_dim=512,
                 n_class=2,
                 dropout=0.1,
                 class_weights=None
                 ):
        super().__init__(config)

        self.config = config
        self.n_class = n_class
        self.bert = RobertaModel(config)
        self.lstm = nn.LSTM(input_size=self.config.hidden_size,
                            hidden_size=h_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)
        if bidirectional:
            h_dim *= 2

        self.projection = ProjectionMLP(in_features=h_dim,
                                        hidden_dim=d_model,
                                        out_features=n_class,
                                        dropout=dropout)
        self.class_weights = class_weights

        # Sum all hidden layers with learnable weights
        # Read more: https://aclanthology.org/2021.eacl-demos.37.pdf
        self.bert_gamma = nn.Parameter(torch.FloatTensor(1, 1))
        self.bert_weights = nn.Parameter(torch.FloatTensor(self.config.num_hidden_layers, 1))

        self.post_init()

    def _init_weights(self, module):
        nn.init.xavier_normal_(self.bert_gamma)
        nn.init.xavier_normal_(self.bert_weights)

    def freeze_transformers(self, num_layers=-1):
        if num_layers == -1:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for layer in self.bert.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        sentence_lengths: Tensor = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.return_dict
        )
        output = bert_output["last_hidden_state"] if self.config.return_dict else bert_output[0]
        if self.config.output_hidden_states:
            all_hidden_layers = bert_output["hidden_states"][1:] if self.config.return_dict else bert_output[-1][1:]
            embeddings = torch.stack([embedding * weight for embedding, weight in zip(all_hidden_layers, self.bert_weights)])
            output = self.bert_gamma * embeddings.sum(dim=0)
        sentence_lengths = sentence_lengths.to("cpu")
        packed_input = pack_padded_sequence(output, sentence_lengths, batch_first=True)
        lstm_out, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.projection(output)

        loss = None
        if labels is not None:
            if self.class_weights is not None and self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=100)
            if isinstance(labels, list):
                labels = torch.tensor([_ for label in labels for _ in label])
            labels = labels.to(logits.device)
            loss = loss_fn(logits.view(-1, self.n_class), labels.view(-1))

        if not self.config.return_dict:
            output = (logits,) + bert_output[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_output.hidden_states,
            attentions=bert_output.attentions,
        )
