import os
import json
from typing import Union, Dict

import torch
from torch import Tensor
from torch import nn

from sentence_transformers.util import fullname, import_from_string


class ProjectionMLP(nn.Module):
    """Feed-forward function with activation function.

    This layer takes a fixed-sized sentence embedding and passes it through a feed-forward layer.
    Can be used to generate deep averaging networks (DAN).

    :param in_features: Size of the input dimension
    :param out_features: Output size
    :param bias: Add a bias vector
    :param activation_function: Pytorch activation function applied on output
    """
    def __init__(self,
                 in_features: int,
                 hidden_dim: int,
                 out_features: int,
                 dropout: float = 0.1,
                 bias: bool = False,
                 affine: bool = False,
                 activation_function=nn.ReLU(inplace=True)):
        super(ProjectionMLP, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.bias = bias
        self.affine = affine
        self.activation_function = activation_function

        # Based on original paper https://arxiv.org/pdf/1502.03167.pdf
        # FC > BN > Activation
        # Activation > BN >
        # list_layers = [nn.Linear(in_features, hidden_dim, bias=bias),
        #                nn.BatchNorm1d(hidden_dim),
        #                activation_function]
        # list_layers += [nn.Linear(hidden_dim, out_features, bias=bias),
        #                 nn.BatchNorm1d(out_features, affine=affine)]

        # https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
        # https://stats.stackexchange.com/questions/526966/using-batchnorm-and-dropout-simultaneously
        # FC > Activation > BN/LN
        list_layers = [nn.Linear(in_features, hidden_dim, bias=bias),
                       nn.LayerNorm(hidden_dim),
                       activation_function,
                       nn.Dropout(dropout),
                       nn.Linear(hidden_dim, out_features, bias=bias)]

        self.net = nn.Sequential(*list_layers)

    def forward(self, features: Union[Dict[str, Tensor], Tensor]):
        if isinstance(features, dict):
            features.update({'sentence_embedding': self.net(features['sentence_embedding'])})
            return features
        else:
            return self.net(features)

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        return {'in_features': self.in_features,
                'hidden_dim': self.hidden_dim,
                'out_features': self.out_features,
                'bias': self.bias,
                'activation_function': fullname(self.activation_function)}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def __repr__(self):
        return "ProjectionMLP({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        config['activation_function'] = import_from_string(config['activation_function'])()
        model = ProjectionMLP(**config)
        model.load_state_dict(
            torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        )
        return model
