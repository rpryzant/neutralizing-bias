import torch.nn as nn
import torch
from torch.autograd import Variable

from tqdm import tqdm

from shared.args import ARGS
from shared.constants import CUDA
from shared.beam import Beam

from seq2seq.utils import dump_outputs



class JointModel(nn.Module):
    def __init__(self, debias_model, tagging_model):
        super(JointModel, self).__init__()
    
        # TODO SHARING EMBEDDINGS FROM DEBIAS
        self.debias_model = debias_model
        self.tagging_model = tagging_model

        self.token_sm = nn.Softmax(dim=2)
        self.time_sm = nn.Softmax(dim=1)
        self.tok_threshold = nn.Threshold(
            ARGS.zero_threshold,
            -10000.0 if ARGS.sequence_softmax else 0.0)


    def run_tagger(self, pre_id, pre_mask, rel_ids=None, pos_ids=None,
                   categories=None):
        _, tok_logits = self.tagging_model(
            pre_id, attention_mask=1.0 - pre_mask, rel_ids=rel_ids,
            pos_ids=pos_ids, categories=categories)

        tok_probs = tok_logits[:, :, :2]
        if ARGS.token_softmax:
            tok_probs = self.token_sm(tok_probs)
        is_bias_probs = tok_probs[:, :, -1]
        is_bias_probs = is_bias_probs.masked_fill(pre_mask, 0.0)

        if ARGS.zero_threshold > -10000.0:
            is_bias_probs = self.tok_threshold(is_bias_probs)

        if ARGS.sequence_softmax:
            is_bias_probs = self.time_sm(is_bias_probs)

        return is_bias_probs, tok_logits


    def forward(self,
            # Debias args.
            pre_id, post_in_id, pre_mask, pre_len, tok_dist,
            # Tagger args.
            rel_ids=None, pos_ids=None, categories=None, ignore_tagger=False):
        global ARGS

        if ignore_tagger:
            is_bias_probs = tok_dist
            tok_logits = None
        else:
            is_bias_probs, tok_logits = self.run_tagger(
                pre_id, pre_mask, rel_ids, pos_ids, categories)

        post_log_probs, post_probs, attns, coverage = self.debias_model(
            pre_id, post_in_id, pre_mask, pre_len, is_bias_probs)

        return post_log_probs, post_probs, is_bias_probs, tok_logits, attns, coverage

    def inference_forward(self,
            # Debias args.
            pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist,
            # Tagger args.
            rel_ids=None, pos_ids=None, categories=None, beam_width=1):
        global CUDA

        if beam_width == 1:
            return self.inference_forward_greedy(
                pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist,
                rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)

        # Encode the source.
        src_outputs, h_t, c_t = self.debias_model.run_encoder(
            pre_id, pre_len, pre_mask)

        # Get the bias probabilities from the tagger.
        is_bias_probs, _ = self.run_tagger(
            pre_id, pre_mask, rel_ids=rel_ids, pos_ids=pos_ids,
            categories=categories)

        # Expand everything per beam. Order is (beam x batch),
        # e.g. [batch, batch, batch] if beam width=3. To unpack, do
        # tensor.view(beam, batch)
        src_outputs = src_outputs.repeat(beam_width, 1, 1)
        initial_hidden = (
            h_t.repeat(beam_width, 1),
            c_t.repeat(beam_width, 1))
        pre_mask = pre_mask.repeat(beam_width, 1)
        pre_len = pre_len.repeat(beam_width)
        if is_bias_probs is not None:
            is_bias_probs = is_bias_probs.repeat(beam_width, 1)

        # Build initial inputs and beams.
        batch_size = pre_id.shape[0]
        beams = [Beam(beam_width, self.debias_model.tok2id, cuda=CUDA)
                 for k in range(batch_size)]

        # Transpose to move beam to first dim.
        tgt_input = torch.stack(
            [b.get_current_state() for b in beams]).t().contiguous().view(
            -1, 1)

        def get_top_hyp():
            out = []
            for b in beams:
                _, ks = b.sort_best()
                hyps = torch.stack([torch.stack(b.get_hyp(k)) for k in ks])
                out.append(hyps)

            # Move beam first. `out` is [beam, batch, len].
            out = torch.stack(out).transpose(1, 0)
            return out

        for i in range(max_len):
            # Run input through the debiasing model.
            with torch.no_grad():
                _, word_probs, _, _ = self.debias_model.run_decoder(
                    pre_id, src_outputs, initial_hidden, tgt_input, pre_mask,
                    is_bias_probs)

            # Tranpose to preserve ordering.
            new_tok_probs = word_probs[:, -1, :].squeeze(1).view(
                beam_width, batch_size, -1).transpose(1, 0)

            for bi in range(batch_size):
                beams[bi].advance(new_tok_probs.data[bi])

            tgt_input = get_top_hyp().contiguous().view(
                batch_size * beam_width, -1)

        return (get_top_hyp()[0].detach().cpu().numpy(),
                is_bias_probs.detach().cpu().numpy())

    def inference_forward_greedy(self,
            # Debias args.
            pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist,
            # Tagger args.
            rel_ids=None, pos_ids=None, categories=None, beam_width=None):
        global CUDA
        global ARGS
        # Initialize target with <s> for every sentence.
        tgt_input = Variable(torch.LongTensor(
            [[post_start_id] for i in range(pre_id.size(0))]))

        if CUDA:
            tgt_input = tgt_input.cuda()

        for i in range(max_len):
            # Run input through the joint model.
            with torch.no_grad():
                _, word_probs, is_bias_probs, _, _, _ = self.forward(
                    pre_id, tgt_input, pre_mask, pre_len, tok_dist,
                    rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)
            next_preds = torch.max(word_probs[:, -1, :], dim=1)[1]
            tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

        # [batch, len] predicted indices.
        return (tgt_input.detach().cpu().numpy(),
                is_bias_probs.detach().cpu().numpy())

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



