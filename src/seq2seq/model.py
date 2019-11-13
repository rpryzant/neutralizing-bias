import math
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel

import sys; sys.path.append('.')
from shared.args import ARGS
from shared.constants import CUDA
import seq2seq.transformer_decoder as transformer
from shared.beam import Beam







class BilinearAttention(nn.Module):
    """ bilinear attention layer: score(H_j, q) = H_j^T W_a q
                (where W_a = self.in_projection)
    """
    def __init__(self, hidden, score_fn='dot'):
        global ARGS
        
        # sanity check that pointer_generator is active
        if ARGS.coverage:
            assert ARGS.pointer_generator

        super(BilinearAttention, self).__init__()
        self.query_in_projection = nn.Linear(hidden, hidden)
        self.key_in_projection = nn.Linear(hidden, hidden)
        # possibly make room for coverage values
        #   (c^t_i  in Eq. 11 of https://arxiv.org/pdf/1704.04368.pdf )
        self.cov_projection = nn.Linear(1, hidden)
        self.softmax = nn.Softmax(dim=1)
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
                ((keys, coverage = [B, L, 1] )  tuple if coverage is set)
            values: [batch, len, hidden] (optional, if none will = keys)
            mask: [batch, len] mask key-scores

            compare query to keys, use the scores to do weighted sum of values
            if no value is specified, then values = keys
        """
        global ARGS
        
        
        if ARGS.coverage:
            keys_in, cov = keys
            att_keys = self.key_in_projection(keys_in) + self.cov_projection(cov)
            
        else:
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
        attn_dists = []
        attn_ctxs = []
        raw_hiddens = []
        output = []
        timesteps = range(input.size(0))
        for i in timesteps:
            hy, cy = self.cell(input[i], hidden)
            if self.use_attention:
                attn_ctx, h_tilde, attn = self.attention_layer(hy, ctx, srcmask)
                hidden = h_tilde, cy
                # most of this is because of the pointer generator stuff...
                output.append(h_tilde)
                attn_dists.append(attn)
                attn_ctxs.append(attn_ctx)
                raw_hiddens.append(hy)
            else:
                hidden = hy, cy
                output.append(hy)

        # combine outputs, and get into [time, batch, dim]
        output = torch.cat(output, 0).view(
            input.size(0), *output[0].size())
        output = output.transpose(0, 1)

        # dists are [time, src len]
        attn_dists = torch.stack(attn_dists).squeeze(1)
        # [time, batch, dim]
        attn_ctxs = torch.stack(attn_ctxs)
        # [time, batch, dim]
        raw_hiddens = torch.stack(raw_hiddens)

        return output, hidden, attn_dists, attn_ctxs, raw_hiddens


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


    def forward(self, input, hidden, ctx, srcmask):
        h_final, c_final = [], []
        for i, layer in enumerate(self.layers):
            #ctx are the attention keys
            output, (h_final_i, c_final_i), attns, attn_ctxs, raw_hiddens = layer(
                input, hidden, ctx, srcmask)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_final.append(h_final_i)
            c_final.append(c_final_i)

        h_final = torch.stack(h_final)
        c_final = torch.stack(c_final)

        # just return top layer attn + raw hiddens
        return input, (h_final, c_final), attns, attn_ctxs, raw_hiddens



class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_dim, dropout, tok2id):
        global ARGS
        global CUDA

        super(Seq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_size
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.pad_id = 0
        self.tok2id = tok2id

        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_id)
        self.encoder = LSTMEncoder(
            self.emb_dim, self.hidden_dim, layers=1, bidirectional=True, dropout=self.dropout)

        self.h_t_projection = nn.Linear(ARGS.hidden_size, ARGS.hidden_size)
        self.c_t_projection = nn.Linear(ARGS.hidden_size, ARGS.hidden_size)

                                                
        self.bridge = nn.Linear(768 if ARGS.bert_encoder else self.hidden_dim, self.hidden_dim)
        
        if ARGS.transformer_decoder:
            self.decoder = transformer.TransformerDecoder(
                num_layers=ARGS.transformer_layers,
                d_model=self.hidden_dim,
                heads=8,
                d_ff=self.hidden_dim,
                copy_attn=False,
                self_attn_type='scaled-dot',
                dropout=self.dropout,
                embeddings=self.embeddings,
                max_relative_positions=0)
        else:
            self.decoder = StackedAttentionLSTM(
                self.emb_dim, self.hidden_dim, layers=1, dropout=self.dropout)

        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)

        # for decoding. TODO -- throw this out?
        self.softmax = nn.Softmax(dim=-1)
        # for training
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

        # pretrained embs from bert (after init to avoid overwrite)
        if ARGS.bert_word_embeddings or ARGS.bert_full_embeddings or ARGS.bert_encoder:
            model = BertModel.from_pretrained(
                'bert-base-uncased',
                cache_dir=ARGS.working_dir + '/cache')

            if ARGS.bert_word_embeddings:
                self.embeddings = model.embeddings.word_embeddings

            if ARGS.bert_encoder:
                self.encoder = model
                # share bert word embeddings with decoder
                self.embeddings = model.embeddings.word_embeddings

            if ARGS.bert_full_embeddings:
                self.embeddings = model.embeddings

        if ARGS.freeze_embeddings:
            for param in self.embeddings.parameters():
                param.requires_grad = False

        # make this even if ARGS.no_tok_enrich so that you can load from
        #   a no-enrichment model (and visa versa)
        self.enrich_input = torch.ones(hidden_size)
        if CUDA:
            self.enrich_input = self.enrich_input.cuda()
        self.enricher = nn.Linear(hidden_size, hidden_size)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def run_encoder(self, pre_id, pre_len, pre_mask):
        global ARGS
        global CUDA
        src_emb = self.embeddings(pre_id)
        if ARGS.bert_encoder:
            # final_hidden_states is [batch_size, sequence_length,
            # hidden_size].
            final_hidden_states, _ = self.encoder(pre_id,
                attention_mask=1.0 - pre_mask, output_all_encoded_layers=False)
            seq_len = final_hidden_states.size()[1]

            # src_outputs is [batch_size, sequence_length, hidden_size].
            src_outputs = self.bridge(final_hidden_states)

            # Average across the sequence length dimension.
            src_h_t = torch.mean(src_outputs, 1)
            src_c_t = torch.mean(src_outputs, 1)

            # Project hidden size to ARGS.hidden_size.
            h_t = self.h_t_projection(src_h_t)
            c_t = self.c_t_projection(src_c_t)

        else:
            src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, pre_len,
                pre_mask)
            src_outputs = self.bridge(src_outputs)
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)

        if ARGS.sigmoid_bridge:
            h_t = nn.Sigmoid()(h_t)
            c_t = nn.Sigmoid()(h_t)

        return src_outputs, h_t, c_t

    def run_decoder(self, pre_id, src_outputs, dec_initial_state, tgt_in_id, pre_mask, tok_dist=None, ignore_enrich=False):
        global ARGS

        # optionally enrich src with tok enrichment
        if not ARGS.no_tok_enrich and not ignore_enrich:
            enrichment = self.enricher(self.enrich_input).repeat(
                src_outputs.shape[0], src_outputs.shape[1], 1)
            enrichment = tok_dist.unsqueeze(2) * enrichment
            src_outputs = src_outputs + enrichment

        tgt_emb = self.embeddings(tgt_in_id)
        tgt_outputs, _, attns, _, _ = self.decoder(tgt_emb, dec_initial_state, src_outputs, pre_mask)

        tgt_outputs_reshape = tgt_outputs.contiguous().view(
            tgt_outputs.size()[0] * tgt_outputs.size()[1],
            tgt_outputs.size()[2])
        logits = self.output_projection(tgt_outputs_reshape)
        logits = logits.view(
            tgt_outputs.size()[0],
            tgt_outputs.size()[1],
            logits.size()[1])

        probs = self.softmax(logits)
        log_probs = self.log_softmax(logits)

        return log_probs, probs, attns, None

    def forward(self, pre_id, post_in_id, pre_mask, pre_len, tok_dist=None, ignore_enrich=False):
        src_outputs, h_t, c_t = self.run_encoder(pre_id, pre_len, pre_mask)
        log_probs, probs, attns, coverage = self.run_decoder(
            pre_id, src_outputs, (h_t, c_t), post_in_id, pre_mask, tok_dist, ignore_enrich)
        return log_probs, probs, attns, coverage


    def inference_forward(self, pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist, beam_width=1):
        global CUDA

        if beam_width == 1:
            return self.inference_forward_greedy(
                pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist)

        # encode src
        src_outputs, h_t, c_t = self.run_encoder(pre_id, pre_len, pre_mask)

        # expand everything per beam. Order is beam x batch,
        #  e.g. [batch, batch, batch] if beam width = 3
        #  so to unpack we do tensor.view(beam, batch)
        src_outputs = src_outputs.repeat(beam_width, 1, 1)
        initial_hidden = (
            h_t.repeat(beam_width, 1),
            c_t.repeat(beam_width, 1))
        pre_mask = pre_mask.repeat(beam_width, 1)
        pre_len = pre_len.repeat(beam_width)
        if tok_dist is not None:
            tok_dist = tok_dist.repeat(beam_width, 1)

        # build initial inputs and beams
        batch_size = pre_id.shape[0]
        beams = [Beam(beam_width, self.tok2id, cuda=CUDA) for k in range(batch_size)]
        # transpose to move beam to first dim
        tgt_input = torch.stack([b.get_current_state() for b in beams]
            ).t().contiguous().view(-1, 1)

        def get_top_hyp():
            out = []
            for b in beams:
                _, ks = b.sort_best()
                hyps = torch.stack([torch.stack(b.get_hyp(k)) for k in ks])
                out.append(hyps)
            # move beam first. output is [beam, batch, len]
            out = torch.stack(out).transpose(1, 0)
            return out

        for i in range(max_len):
            # run input through the model
            with torch.no_grad():
                _, word_probs, _, _ = self.run_decoder(
                    pre_id, src_outputs, initial_hidden, tgt_input, pre_mask, tok_dist)
            # transpose to preserve ordering
            new_tok_probs = word_probs[:, -1, :].squeeze(1).view(
                beam_width, batch_size, -1).transpose(1, 0)

            for bi in range(batch_size):
                beams[bi].advance(new_tok_probs.data[bi])

            tgt_input = get_top_hyp().contiguous().view(batch_size * beam_width, -1)

        return get_top_hyp()[0].detach().cpu().numpy()


    def inference_forward_greedy(self, pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist):
        global CUDA
        """ argmax decoding """
        # Initialize target with <s> for every sentence
        tgt_input = Variable(torch.LongTensor([
                [post_start_id] for i in range(pre_id.size(0))
        ]))
        if CUDA:
            tgt_input = tgt_input.cuda()

        for i in range(max_len):
            # run input through the model
            with torch.no_grad():
                _, word_probs, _, _ = self.forward(
                    pre_id, tgt_input, pre_mask, pre_len, tok_dist)
            next_preds = torch.max(word_probs[:, -1, :], dim=1)[1]
            tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

        # [batch, len ] predicted indices
        return tgt_input.detach().cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class PointerSeq2Seq(Seq2Seq):
    """ https://arxiv.org/pdf/1704.04368.pdf """
    def __init__(self, vocab_size, hidden_size, emb_dim, dropout, tok2id):
        global CUDA
        global ARGS

        super(PointerSeq2Seq, self).__init__(
            vocab_size, hidden_size, emb_dim, dropout, tok2id)

        # 768 = input (embedding) size
        # TODO make this a constant, maybe in args?s
        self.p_gen_W = nn.Linear((hidden_size * 3) + 768, 1)
        self.p_gen_sigmoid = nn.Sigmoid()

    def run_decoder(self, pre_id, src_outputs, dec_initial_state, tgt_in_id, pre_mask, tok_dist=None, ignore_enrich=False):
        global ARGS
        global CUDA

        # optionally enrich src with tok enrichment
        if not ARGS.no_tok_enrich and not ignore_enrich:
            enrichment = self.enricher(self.enrich_input).repeat(
                src_outputs.shape[0], src_outputs.shape[1], 1)
            enrichment = tok_dist.unsqueeze(2) * enrichment
            src_outputs = src_outputs + enrichment

        # initialize inputs, hidden states, counters, etc
        tgt_emb = self.embeddings(tgt_in_id)
        tgt_output_probs = []
        attns = []
        hidden = dec_initial_state

        if ARGS.coverage:
            coverage_vecs = []
            coverage = torch.zeros(src_outputs.shape[:2])
            if CUDA:
                coverage = coverage.cuda()

        # manually crank the decoder
        for ti in range(tgt_emb.shape[1]):
            # pull out decoder input for timestep i
            tgt_emb_i = tgt_emb[:, ti, :].unsqueeze(1)

            if ARGS.coverage:
                # add coverage values to attention inputs
                attn_keys = (src_outputs, coverage.unsqueeze(-1).clone())
            else:
                attn_keys = src_outputs

            # run decoder on this step
            output_i, (h_tilde_i, ci), attn, attn_ctx, raw_hidden = self.decoder(
                tgt_emb_i, hidden, attn_keys, pre_mask)
            output_i = output_i.squeeze(1)
            attn = attn.squeeze(0)
            h_i = raw_hidden.squeeze(0)

            # accumulate attention scores
            if ARGS.coverage:
                coverage += attn
                coverage_vecs.append(coverage.clone())

            # get probability of generating vs copying
            p_gen = self.p_gen_W(torch.cat([
                attn_ctx.squeeze(0), h_i,
                ci.squeeze(0), tgt_emb_i.squeeze(1)
            ], -1))
            p_gen = self.p_gen_sigmoid(p_gen)

            # final output distribution is
            #        ( p_gen * pred_dist )  +  ((1 - p_gen) * attn_dist)
            gen_probs = p_gen * self.softmax(self.output_projection(output_i))
            pointer_probs = (1 - p_gen) * attn
            gen_probs.scatter_add_(1, pre_id, pointer_probs)

            # update counters and hidden state
            tgt_output_probs.append(gen_probs)
            attns.append(attn)
            hidden = (h_tilde_i.squeeze(0), ci.squeeze(0))

        probs = torch.stack(tgt_output_probs)
        probs = probs.permute(1, 0, 2)

        log_probs = torch.log(probs)

        return (
            log_probs, 
            probs, 
            torch.stack(attns), 
            torch.stack(coverage_vecs) if ARGS.coverage else None
        )
