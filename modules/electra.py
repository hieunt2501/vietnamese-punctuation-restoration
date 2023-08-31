from typing import Optional, List

import torch
from torchcrf import CRF
from transformers import ElectraForTokenClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class CrfSequenceClassifierOutput(SequenceClassifierOutput):
    sequence_tags: Optional[List[List[int]]]


class PuncElectraCrfModel(ElectraForTokenClassification):
    def __init__(self, config):
        super(PuncElectraCrfModel, self).__init__(config=config)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None, valid_ids=None,
                label_mask=None, **kwargs):
        electra_output = self.electra(input_ids, attention_mask, head_mask=None)
        sequence_output = electra_output[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=input_ids.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j] == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        logits_mask = valid_output.sum(-1) != 0
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        loss = None
        sequence_tags = torch.tensor(self.crf.decode(logits), device=input_ids.device)
        sequence_tags = sequence_tags * logits_mask

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_mask.type(torch.bool))
            loss = -1.0 * log_likelihood

        if not self.config.return_dict:
            output = (logits, sequence_tags) + sequence_output[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=sequence_tags,
            hidden_states=electra_output.hidden_states,
            attentions=electra_output.attentions,
        )