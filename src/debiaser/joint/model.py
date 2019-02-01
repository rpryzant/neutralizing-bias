import torch.nn as nn
import torch
from torch.autograd import Variable

from seq2seq_utils import dump_outputs

from tqdm import tqdm

from joint_args import ARGS



CUDA = (torch.cuda.device_count() > 0)


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
            pre_id, post_start_id, pre_mask, pre_len, tok_dist, type_id, ignore_enrich=False,   # debias arggs
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
                decoder_logit, word_probs = self.forward(
                    pre_id, tgt_input, pre_mask, pre_len, tok_dist, type_id,
                    rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)
            next_preds = torch.max(word_probs[:, -1, :], dim=1)[1]
            tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

        # [batch, len ] predicted indices
        return tgt_input.detach().cpu().numpy()
                
    def forward(self, 
        pre_id, post_in_id, pre_mask, pre_len, tok_dist, type_id, ignore_enrich=False,   # debias arggs
        rel_ids=None, pos_ids=None, categories=None):      # tagging args
        global ARGS

        if rel_ids is None or pos_ids is None:
            is_bias_probs = tok_dist
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
            pre_id, post_in_id, pre_mask, pre_len,  is_bias_probs, type_id)

        return post_logits, post_probs



def run_eval(model, dataloader, tok2id, out_file_path, max_seq_len, beam_width=1):
    id2tok = {x: tok for (tok, x) in tok2id.items()}

    out_file = open(out_file_path, 'w')

    losses = []
    hits = []
    preds, golds, srcs = [], [], []
    for step, batch in enumerate(tqdm(dataloader)):
        # if step > 1: continue
    
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, _, tok_dist,
            replace_id, rel_ids, pos_ids, type_id, categories
        ) = batch

        post_start_id = tok2id['行']
        max_len = min(max_seq_len, pre_len[0].detach().cpu().numpy() + 10)

        with torch.no_grad():
            predicted_toks = model.inference_forward(
                pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist, type_id,
                rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)

        new_hits, new_preds, new_golds, new_srcs = dump_outputs(
            pre_id.detach().cpu().numpy(), 
            post_out_id.detach().cpu().numpy(), 
            predicted_toks, 
            replace_id.detach().cpu().numpy(), 
            pre_tok_label_id.detach().cpu().numpy(), 
            id2tok, out_file)
        hits += new_hits
        preds += new_preds
        golds += new_golds
        srcs += new_srcs
    out_file.close()

    return hits, preds, golds, srcs





