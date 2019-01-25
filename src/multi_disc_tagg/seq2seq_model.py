import math
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel

from seq2seq_args import ARGS

CUDA = (torch.cuda.device_count() > 0)



def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


"""Beam search implementation in PyTorch."""
# Takes care of beams, back pointers, and scores.
# Borrowed from OpenNMT
class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, tok2id, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = tok2id['[PAD]']
        self.bos = tok2id['行']
        self.eos = tok2id['止']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step. [time, beam]
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos # TODO CHANGED THIS

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # -2 to include start tok
        for j in range(len(self.prevKs) - 1, -2, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]









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
    def __init__(self, vocab_size, hidden_size, emb_dim, dropout, tok2id):
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

                                                
        self.bridge = nn.Linear(768 if ARGS.bert_encoder else self.hidden_dim, self.hidden_dim)
        
        self.decoder = StackedAttentionLSTM(
            self.emb_dim, self.hidden_dim, layers=1, dropout=self.dropout)
        
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

        # pretrained embs from bert (after init to avoid overwrite)
        if ARGS.bert_word_embeddings or ARGS.bert_full_embeddings or ARGS.bert_encoder:
            model = BertModel.from_pretrained(
                'bert-base-uncased',
                ARGS.working_dir + '/cache')
                
            if ARGS.bert_word_embeddings:
                self.embeddings = copy.deepcopy(model.embeddings.word_embeddings)
                
            if ARGS.bert_full_embeddings:
                self.embeddings = copy.deepcopy(model.embeddings)

            if ARGS.bert_encoder:
                self.encoder = model
                # share bert word embeddings with decoder
                self.embeddings = model.embeddings.word_embeddings

        if ARGS.freeze_embeddings:
            for param in self.embeddings.parameters():
                param.requires_grad = False


    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)
    
    def run_encoder(self, pre_id, pre_len, pre_mask):
        src_emb = self.embeddings(pre_id)
        src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, pre_len, pre_mask)
        src_outputs = self.bridge(src_outputs)
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)

        return src_outputs, h_t, c_t

    def run_decoder(self, src_outputs, dec_initial_state, tgt_in_id, pre_mask, tok_dist=None, type_id=None):
        tgt_emb = self.embeddings(tgt_in_id)
        tgt_outputs, _ = self.decoder(tgt_emb, dec_initial_state, src_outputs, pre_mask)

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

    def forward(self, pre_id, post_in_id, pre_mask, pre_len, tok_dist=None, type_id=None, ignore_enrich=False):
        src_outputs, h_t, c_t = self.run_encoder(pre_id, pre_len, pre_mask)
        logits, probs = self.run_decoder(
            src_outputs, (h_t, c_t), post_in_id, pre_mask)

        return logits, probs


    def inference_forward(self, pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist, type_id, beam_width=1):
        global CUDA

        if beam_width == 1:
            return self.inference_forward_greedy(
                pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist, type_id)

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
        if type_id is not None:
            type_id = type_id.repeat(beam_width)

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
                decoder_logit, word_probs = self.run_decoder(
                    src_outputs, initial_hidden, tgt_input, pre_mask, tok_dist, type_id)
            # tranpose to preserve ordering
            new_tok_probs = word_probs[:, -1, :].squeeze(1).view(
                beam_width, batch_size, -1).transpose(1, 0)

            for bi in range(batch_size):
                beams[bi].advance(new_tok_probs.data[bi])

            tgt_input = get_top_hyp().contiguous().view(batch_size * beam_width, -1)

        return get_top_hyp()[0].detach().cpu().numpy()


    def inference_forward_greedy(self, pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist, type_id):
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
                decoder_logit, word_probs = self.forward(pre_id, tgt_input, pre_mask, pre_len, tok_dist, type_id)
            next_preds = torch.max(word_probs[:, -1, :], dim=1)[1]
            tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)
            # move to cpu because otherwise quickly runs out of mem
#            out_logits.append(decoder_logit[:, -1, :].detach().cpu())

        # [batch, len ] predicted indices
        return tgt_input.detach().cpu().numpy()#, torch.stack(out_logits).permute(1, 0, 2)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Seq2SeqEnrich(Seq2Seq):
    def __init__(self, vocab_size, hidden_size, emb_dim, dropout, tok2id):
        global CUDA
        super(Seq2SeqEnrich, self).__init__(
            vocab_size, hidden_size, emb_dim, dropout, tok2id)

        # i dont think i need to do this but just to be safe...
        self.enrich_input0 = torch.ones(hidden_size)
        self.enrich_input1 = torch.ones(hidden_size)
        if CUDA:
            self.enrich_input0 = self.enrich_input0.cuda()
            self.enrich_input1 = self.enrich_input1.cuda()

        # seperate enrichers for del/edit
        self.enricher0 = nn.Linear(hidden_size, hidden_size)
        self.enricher1 = nn.Linear(hidden_size, hidden_size)
        # # because init_weights was called in super and dont want to fuq up embeddings
        # self.enricher.weight.data.uniform_(-0.1, 0.1)  


    def run_decoder(self, src_outputs, dec_initial_state, tgt_in_id, pre_mask, tok_dist=None, type_id=None, ignore_enrich=False):
        global ARGS
        # make a "change this token" embedding and add it to the
        # src_output token that should be changed
        if ARGS.fine_enrichment:
            enrichment0 = self.enricher0(self.enrich_input0).repeat(
                src_outputs.shape[0], src_outputs.shape[1], 1)
            enrichment1 = self.enricher1(self.enrich_input1).repeat(
                src_outputs.shape[0], src_outputs.shape[1], 1)
            type_id = type_id.unsqueeze(1).unsqueeze(2)
            enrichment = (enrichment1 * type_id) + (enrichment0 * (1 - type_id))
        else:
            enrichment = self.enricher0(self.enrich_input0).repeat(
                src_outputs.shape[0], src_outputs.shape[1], 1)
        enrichment = tok_dist.unsqueeze(2) * enrichment

        if not ignore_enrich:
            src_outputs = src_outputs + enrichment

        tgt_emb = self.embeddings(tgt_in_id)
        tgt_outputs, _ = self.decoder(tgt_emb, dec_initial_state, src_outputs, pre_mask)

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


    def forward(self, pre_id, post_in_id, pre_mask, pre_len, tok_dist, type_id, ignore_enrich=False):
        src_outputs, h_t, c_t = self.run_encoder(pre_id, pre_len, pre_mask)
        logits, probs = self.run_decoder(
            src_outputs, (h_t, c_t), post_in_id, pre_mask, tok_dist, type_id, ignore_enrich)
        
        return logits, probs
        

