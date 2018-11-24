"""Sequence to Sequence models."""
import glob
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import decoders
import encoders

from cuda import CUDA


def get_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    # nothing to load, continue with fresh params
    if len(ckpts) == 0:
        return -1, None
    ckpts = map(lambda ckpt: (
        int(ckpt.split('.')[1]),
        ckpt), ckpts)
    # get most recent checkpoint
    epoch, ckpt_path = sorted(ckpts)[-1]
    return epoch, ckpt_path


def attempt_load_model(model, checkpoint_dir=None, checkpoint_path=None):
    assert checkpoint_dir or checkpoint_path

    if checkpoint_dir:
        epoch, checkpoint_path = get_latest_ckpt(checkpoint_dir)
    else:
        epoch = int(checkpoint_path.split('.')[-2])

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print('Load from %s sucessful!' % checkpoint_path)
        return model, epoch + 1
    else:
        return model, 0


class SeqModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        pad_id_src,
        pad_id_tgt,
        config=None,
    ):
        """Initialize model."""
        super(SeqModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt
        self.batch_size = config['data']['batch_size']
        self.config = config
        self.options = config['model']
        self.model_type = config['model']['model_type']

        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.options['emb_dim'],
            self.pad_id_src)

        if self.config['data']['share_vocab']:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(
                self.tgt_vocab_size,
                self.options['emb_dim'],
                self.pad_id_tgt)

        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'])
            self.ctx_bridge = nn.Linear(
                self.options['src_hidden_dim'],
                self.options['tgt_hidden_dim'])

        else:
            raise NotImplementedError('unknown encoder type')

        # # # # # #  # # # # # #  # # # # #  NEW STUFF FROM STD SEQ2SEQ
        
        if self.model_type == 'delete':
            self.attribute_embedding = nn.Embedding(
                num_embeddings=2, 
                embedding_dim=self.options['emb_dim'])
            attr_size = self.options['emb_dim']

        elif self.model_type == 'delete_retrieve':
            self.attribute_encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'],
                pack=False)
            attr_size = self.options['src_hidden_dim']

        elif self.model_type == 'seq2seq':
            attr_size = 0

        else:
            raise NotImplementedError('unknown model type')

        self.c_bridge = nn.Linear(
            attr_size + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])
        self.h_bridge = nn.Linear(
            attr_size + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])

        # # # # # #  # # # # # #  # # # # # END NEW STUFF

        self.decoder = decoders.StackedAttentionLSTM(config=config)

        self.output_projection = nn.Linear(
            self.options['tgt_hidden_dim'],
            tgt_vocab_size)

        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.h_bridge.bias.data.fill_(0)
        self.c_bridge.bias.data.fill_(0)
        self.output_projection.bias.data.fill_(0)

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask):
        src_emb = self.src_embedding(input_src)

        srcmask = (1-srcmask).byte()

        src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, srclens, srcmask)

        if self.options['bidirectional']:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        src_outputs = self.ctx_bridge(src_outputs)


        # # # #  # # # #  # #  # # # # # # #  # # seq2seq diff
        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above

        if self.model_type == 'delete':
            # just do h i guess?
            a_ht = self.attribute_embedding(input_attr)
            c_t = torch.cat((c_t, a_ht), -1)
            h_t = torch.cat((h_t, a_ht), -1)

        elif self.model_type == 'delete_retrieve':
            attr_emb = self.src_embedding(input_attr)
            _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attrlens, attrmask)
            if self.options['bidirectional']:
                a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
                a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
            else:
                a_ht = a_ht[-1]
                a_ct = a_ct[-1]

            h_t = torch.cat((h_t, a_ht), -1)
            c_t = torch.cat((c_t, a_ct), -1)
            
        c_t = self.c_bridge(c_t)
        h_t = self.h_bridge(h_t)

        # # # #  # # # #  # #  # # # # # # #  # # end diff

        tgt_emb = self.tgt_embedding(input_tgt)
        tgt_outputs, (_, _) = self.decoder(
            tgt_emb,
            (h_t, c_t),
            src_outputs,
            srcmask)

        tgt_outputs_reshape = tgt_outputs.contiguous().view(
            tgt_outputs.size()[0] * tgt_outputs.size()[1],
            tgt_outputs.size()[2])
        decoder_logit = self.output_projection(tgt_outputs_reshape)
        decoder_logit = decoder_logit.view(
            tgt_outputs.size()[0],
            tgt_outputs.size()[1],
            decoder_logit.size()[1])

        probs = self.softmax(decoder_logit)

        return decoder_logit, probs

    def count_params(self):
        n_params = 0
        for param in self.parameters():
            n_params += np.prod(param.data.cpu().numpy().shape)
        return n_params
        
        
        
