import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


import ops



class AttentionalLSTM(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_dim, hidden_dim, config, attention):
        """Initialize params."""
        super(AttentionalLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.use_attention = attention
        self.config = config
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

        if self.use_attention:
            self.attention_layer = ops.BilinearAttention(hidden_dim)


    def forward(self, input, hidden, ctx, srcmask, kb=None):
        input = input.transpose(0, 1)

        output = []
        timesteps = range(input.size(0))
        for i in timesteps:
            hy, cy = self.cell(input[i], hidden)
            if self.use_attention:
                _, h_tilde, alpha = self.attention_layer(hy, ctx, srcmask)
                hidden = h_tilde, cy
                output.append(h_tilde)
            else: 
                hidden = hy, cy
                output.append(hy)

        # combine outputs, and get into [time, batch, dim]
        output = torch.cat(output, 0).view(
            input.size(0), *output[0].size())

        output = output.transpose(0, 1)

        return output, hidden


class StackedAttentionLSTM(nn.Module):
    """ stacked lstm with input feeding
    """
    def __init__(self, cell_class=AttentionalLSTM, config=None):
        super(StackedAttentionLSTM, self).__init__()
        self.options=config['model']


        self.dropout = nn.Dropout(self.options['dropout'])

        self.layers = []
        input_dim = self.options['emb_dim']
        hidden_dim = self.options['tgt_hidden_dim']
        for i in range(self.options['tgt_layers']):
            layer = cell_class(input_dim, hidden_dim, config, config['model']['attention'])
            self.add_module('layer_%d' % i, layer)
            self.layers.append(layer)
            input_dim = hidden_dim


    def forward(self, input, hidden, ctx, srcmask, kb=None):
        h_final, c_final = [], []
        for i, layer in enumerate(self.layers):
            output, (h_final_i, c_final_i) = layer(input, hidden, ctx, srcmask, kb)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_final.append(h_final_i)
            c_final.append(c_final_i)

        h_final = torch.stack(h_final)
        c_final = torch.stack(c_final)

        return input, (h_final, c_final)


