import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


CUDA = (torch.cuda.device_count() > 0)

class BilinearAttention(nn.Module):
    """ bilinear attention layer: score(H_j, q) = H_j^T W_a q
                (where W_a = self.in_projection)
    """
    def __init__(self, hidden, score_fn='dot'):
        super(BilinearAttention, self).__init__()
        self.query_in_projection = nn.Linear(hidden, hidden)
        self.key_in_projection = nn.Linear(hidden, hidden)
        self.softmax = nn.Softmax()
        self.out_projection = nn.Linear(hidden * 2, hidden)
        self.tanh = nn.Tanh()
        self.score_fn = self.dot

        if score_fn == 'bahdanau':
            self.v_att = nn.Linear(hidden, 1, bias=False)
            self.score_tanh = nn.Tanh()
            self.score_fn = self.bahdanau

    def forward(self, query, keys, mask=None, values=None):
        """
            query: [batch, hidden]
            keys: [batch, len, hidden]
            values: [batch, len, hidden] (optional, if none will = keys)
            mask: [batch, len] mask key-scores

            compare query to keys, use the scores to do weighted sum of values
            if no value is specified, then values = keys
        """
        att_keys = self.key_in_projection(keys)

        if values is None:
            values = att_keys

        # [Batch, Hidden, 1]
        att_query = self.query_in_projection(query)

        # [Batch, Source length]
        attn_scores = self.score_fn(att_keys, att_query)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -float('inf'))

        attn_probs = self.softmax(attn_scores)

        # [Batch, 1, source length]
        attn_probs_transposed = attn_probs.unsqueeze(1)
        # [Batch, hidden]
        weighted_context = torch.bmm(attn_probs_transposed, values).squeeze(1)

        context_query_mixed = torch.cat((weighted_context, query), 1)
        context_query_mixed = self.tanh(self.out_projection(context_query_mixed))

        return weighted_context, context_query_mixed, attn_probs


    def dot(self, keys, query):
        """
        keys: [B, T, H]
        query: [B, H]
        """
        return torch.bmm(keys, query.unsqueeze(2)).squeeze(2)


    def bahdanau(self, keys, query):
        """
        keys: [B, T, H]
        query: [B, H]
        """
        return self.v_att(self.score_tanh(keys + query.unsqueeze(1))).squeeze(2)


class LSTMEncoder(nn.Module):
    """ simple wrapper for a bi-lstm """
    def __init__(self, emb_dim, hidden_dim, layers, bidirectional, dropout, pack=True):
        super(LSTMEncoder, self).__init__()

        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim // self.num_directions,
            layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout)

        self.pack = pack

    def init_state(self, batch_size):
        global CUDA

        h0 = Variable(torch.zeros(
            self.lstm.num_layers * self.num_directions,
            batch_size,
            self.lstm.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.lstm.num_layers * self.num_directions,
            batch_size,
            self.lstm.hidden_size
        ), requires_grad=False)

        if CUDA:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0


    def forward(self, src_embedding, srclens, srcmask):
        # retrieve batch size dynamically for decoding
        h0, c0 = self.init_state(batch_size=src_embedding.size(0))

        if self.pack:
            inputs = pack_padded_sequence(src_embedding, srclens, batch_first=True)
        else:
            inputs = src_embedding

        outputs, (h_final, c_final) = self.lstm(inputs, (h0, c0))

        if self.pack:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        return outputs, (h_final, c_final)



class AttentionalLSTM(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_dim, hidden_dim, use_attention=True):
        """Initialize params."""
        super(AttentionalLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.use_attention = use_attention
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

        if self.use_attention:
            self.attention_layer = BilinearAttention(hidden_dim)


    def forward(self, input, hidden, ctx, srcmask):
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
    def __init__(self, emb_dim, hidden_dim, layers, dropout, cell_class=AttentionalLSTM):
        super(StackedAttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.layers = []

        for i in range(layers):
            layer = cell_class(emb_dim, hidden_dim)
            self.add_module('layer_%d' % i, layer)
            self.layers.append(layer)
            input_dim = hidden_dim


    def forward(self, input, hidden, ctx, srcmask):
        h_final, c_final = [], []
        for i, layer in enumerate(self.layers):
            output, (h_final_i, c_final_i) = layer(input, hidden, ctx, srcmask)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_final.append(h_final_i)
            c_final.append(c_final_i)

        h_final = torch.stack(h_final)
        c_final = torch.stack(c_final)

        return input, (h_final, c_final)



class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_dim, dropout):
        super(Seq2Seq, self).__init__()        

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_size
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.pad_id = 0
        
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_id)

        self.encoder = LSTMEncoder(
            self.emb_dim, self.hidden_dim, layers=1, bidirectional=True, dropout=self.dropout)
        self.ctx_bridge = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.decoder = StackedAttentionLSTM(
            self.emb_dim, self.hidden_dim, layers=1, dropout=self.dropout)
        
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.init_weights()
        
        
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)
    
    
    def forward(self, pre_id, post_in_id, pre_mask, pre_len):
        src_emb = self.embeddings(pre_id)
        
        src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, pre_len, pre_mask)
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)

        tgt_emb = self.embeddings(post_in_id)

        tgt_outputs, _ = self.decoder(tgt_emb, (h_t, c_t), src_outputs, pre_mask)

        tgt_outputs_reshape = tgt_outputs.contiguous().view(
            tgt_outputs.size()[0] * tgt_outputs.size()[1],
            tgt_outputs.size()[2])
        logits = self.output_projection(tgt_outputs_reshape)
        logits = logits.view(
            tgt_outputs.size()[0],
            tgt_outputs.size()[1],
            logits.size()[1])

        probs = self.softmax(logits)
        
        return logits, probs


    def inference_forward(self, pre_id, post_start_id, pre_mask, pre_len, max_len):
        global CUDA
        """ argmax decoding """
        # Initialize target with <s> for every sentence
        tgt_input = Variable(torch.LongTensor([
                [post_start_id] for i in range(pre_id.size(0))
        ]))
        if CUDA:
            tgt_input = tgt_input.cuda()

        out_logits = []

        for i in range(max_len):
            # run input through the model
            with torch.no_grad():
                decoder_logit, word_probs = self.forward(pre_id, tgt_input, pre_mask, pre_len)
            next_preds = torch.max(word_probs, dim=2)[1][:, -1]
            tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

            # move to cpu because otherwise quickly runs out of mem
#            out_logits.append(decoder_logit[:, -1, :].detach().cpu())

        return tgt_input#, torch.stack(out_logits).permute(1, 0, 2)
        
        
        
