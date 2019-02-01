"""Sequence to Sequence models."""
import glob
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn import svm
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

import decoders
import encoders
import ops

from cuda import CUDA

from simplediff import diff


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

        elif self.model_type in 'delete_retrieve':
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
            self.options['src_hidden_dim'],
            self.options['tgt_hidden_dim'])
        self.h_bridge = nn.Linear(
            attr_size + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])

        if self.config['experimental']['predict_sides']:
            if self.config['experimental']['side_attn_type'] == 'feedforward':
                self.side_attn = ops.FeedForwardAttention(
                    input_dim=self.options['src_hidden_dim'],
                    hidden_dim=self.options['src_hidden_dim'],
                    layers=2,
                    dropout=self.options['dropout'])
            elif self.config['experimental']['side_attn_type'] == 'dot':
                self.side_attn = ops.BilinearAttention(
                    hidden=self.options['src_hidden_dim'])
            elif self.config['experimental']['side_attn_type'] == 'bahdanau':
                self.side_attn = ops.BilinearAttention(
                    hidden=self.options['src_hidden_dim'],
                    score_fn='bahdanau')

            self.side_predictor = ops.FFNN(
                input_dim=self.options['src_hidden_dim'],
                hidden_dim=self.options['src_hidden_dim'],
                output_dim=self.config['experimental']['n_side_outputs'],    # TODO -- SET SOMEWHERE
                nlayers=2,
                dropout=self.options['dropout']) 

            if self.config['experimental']['add_side_embeddings']:
                self.side_embeddings = nn.Parameter(
                    torch.zeros(self.config['experimental']['n_side_outputs'], self.options['emb_dim']),
                    requires_grad=True)
                self.h_compression = nn.Linear(
                    self.options['emb_dim'] + self.options['src_hidden_dim'], 
                    self.options['tgt_hidden_dim'])
                self.side_softmax = nn.Softmax(dim=-1)

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
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask, side_info):
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
        if self.config['experimental']['predict_sides']:
            side_info = side_info[:, 1:].squeeze(1)  # ignore the "start token" from data.get_minibatch
            query = torch.zeros(src_outputs[:, 0, :].shape)
            if CUDA:
                query = query.cuda()
            src_summary, _, probs = self.side_attn(
                query=query,
                keys=src_outputs, 
                values=src_outputs,
                mask=srcmask)

            side_logit, side_loss = self.side_predictor(
                src_summary, side_info)
            if self.config['experimental']['add_side_embeddings']:
                # use probs to do weighted sum of embeddings, join those with h_T
                probs = self.side_softmax(side_logit)
                if self.config['experimental']['side_embedding_teacher_force']:
                    probs = torch.zeros(probs.shape)
                    if CUDA:
                        probs = probs.cuda()
                    probs = probs.scatter_(1, side_info.unsqueeze(1), 1.0)
                embs = self.side_embeddings.repeat(side_logit.shape[0], 1, 1)
                weighted_emb = torch.bmm(probs.unsqueeze(1), embs).squeeze(1)
                h_t = torch.cat((h_t, weighted_emb), -1)
                h_t = self.h_compression(h_t)
        else:
            side_logit, side_loss = None, 0.0

        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above?
        if self.model_type == 'delete':
            # just do h i guess?
            a_ht = self.attribute_embedding(input_attr)
            h_t = torch.cat((h_t, a_ht), -1)

        elif self.model_type == 'delete_retrieve':
            attr_emb = self.src_embedding(input_attr)
            _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attrlens, attrmask)
            if self.options['bidirectional']:
                a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
            else:
                a_ht = a_ht[-1]

            h_t = torch.cat((h_t, a_ht), -1)
            
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

        return decoder_logit, probs, side_logit, side_loss

    def count_params(self):
        n_params = 0
        for param in self.parameters():
            n_params += np.prod(param.data.cpu().numpy().shape)
        return n_params
        
        
        




class TextClassifier(object):
    def __init__(self, vocab=None):
        # vocab: {tok: idx}
        if vocab:
            self.vectorizer = CountVectorizer(vocabulary=vocab, binary=True)
        else:
            self.vectorizer = None

        self.predictor = svm.LinearSVC()

    def parameters(self):
        id2tok = {i: x for x, i in self.vectorizer.vocabulary_.items()}
        out = {id2tok[i]: coef for i, coef in enumerate(self.predictor.coef_[0])}
        return out

    def fit(self, corpus1_path, corpus2_path, seed=0):
        sents = [x.strip() for x in open(corpus1_path)] + [x.strip() for x in open(corpus2_path)]

        X = self.vectorizer.fit_transform(sents)
        Y = [0 for _ in open(corpus1_path)] + [1 for _ in open(corpus2_path)]
        X, Y = shuffle(X, Y, random_state=seed)

        self.predictor.fit(X, Y)
        
    def predict(self, seqs):
        X = self.vectorizer.transform(seqs)
        Y_hat = self.predictor.predict(X)
        return Y_hat
        
    def error_rate(self, seqs, Y):
        Y_hat = self.predict(seqs)
        error = len([yhat for yhat, y in zip(Y_hat, Y) if yhat != y]) * 1.0 / len(Y)
        return error
        
    def save(self, path_prefix):
        with open(path_prefix + '.vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(path_prefix + '.predictor.pkl', 'wb') as f:
            pickle.dump(self.predictor, f)

    def load(self, path_prefix):
        with open(path_prefix + '.vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(path_prefix + '.predictor.pkl', 'rb') as f:
            self.predictor = pickle.load(f)

    @staticmethod
    def from_pickle(path_prefex):
        out = TextClassifier()
        out.load(path_prefex)
        return out



