import torch.nn as nn
import torch
from torch.autograd import Variable

from tqdm import tqdm

from shared.args import ARGS
from shared.constants import CUDA

from seq2seq.utils import dump_outputs

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

        post_log_probs, post_probs = self.debias_model(
            pre_id, post_in_id, pre_mask, pre_len, is_bias_probs)

        return post_log_probs, post_probs, is_bias_probs, tok_logits

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
                _, word_probs = self.debias_model.run_decoder(
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
                _, word_probs, is_bias_probs, _ = self.forward(
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



