import torch.nn as nn
import torch
from torch.autograd import Variable

from tqdm import tqdm

from shared.args import ARGS
from shared.constants import CUDA

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

    def inference_forward(self,
            pre_id, post_start_id, pre_mask, pre_len, tok_dist, ignore_enrich=False,   # debias arggs
                          rel_ids=None, pos_ids=None, categories=None, beam_width=None):      # tagging args
        global CUDA
        global ARGS
        """ argmax decoding """
        # Initialize target with <s> for every sentence
        tgt_input = Variable(torch.LongTensor([
                [post_start_id] for i in range(pre_id.size(0))
        ]))
        if CUDA:
            tgt_input = tgt_input.cuda()

        out_logits = []

        for i in range(ARGS.max_seq_len):
            # run input through the model
            with torch.no_grad():
                decoder_logit, word_probs, tok_probs, _ = self.forward(
                    pre_id, tgt_input, pre_mask, pre_len, tok_dist,
                    rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)
            next_preds = torch.max(word_probs[:, -1, :], dim=1)[1]
            tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

        # [batch, len ] predicted indices
        return tgt_input.detach().cpu().numpy(), tok_probs.detach().cpu().numpy()
                
    def forward(self, 
        pre_id, post_in_id, pre_mask, pre_len, tok_dist, ignore_enrich=False,   # debias arggs
        rel_ids=None, pos_ids=None, categories=None, ignore_tagger=False):      # tagging args
        global ARGS

        if ignore_tagger:
            is_bias_probs = tok_dist
            tok_logits = None
        else:
            category_logits, tok_logits = self.tagging_model(
                pre_id, attention_mask=1.0-pre_mask, rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)

            tok_probs = tok_logits[:, :, :2]
            if ARGS.token_softmax:
                tok_probs = self.token_sm(tok_probs)
            is_bias_probs = tok_probs[:, :, -1]

            if ARGS.zero_threshold > -10000.0:
                is_bias_probs = self.tok_threshold(is_bias_probs)

            if ARGS.sequence_softmax:
                is_bias_probs = self.time_sm(is_bias_probs)

        post_logits, post_probs = self.debias_model(
            pre_id, post_in_id, pre_mask, pre_len,  is_bias_probs)

        return post_logits, post_probs, is_bias_probs, tok_logits


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



