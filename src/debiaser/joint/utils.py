from tqdm import tqdm

import torch.nn as nn
import torch

import sys; sys.path.append(".")
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.utils as seq2seq_utils




def train_for_epoch(model, dataloader, optimizer, debias_loss_fn, tagging_loss_fn=None, ignore_tagger=False, coverage=False):
    model.train()
    losses = []
    for step, batch in enumerate(tqdm(dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA: 
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, post_tok_label_id,
            rel_ids, pos_ids, categories
        ) = batch      
        post_log_probs, post_probs, tok_probs, tok_logits, attns, coverages = model(
            pre_id, post_in_id, pre_mask, pre_len, pre_tok_label_id,
            rel_ids=rel_ids, pos_ids=pos_ids, categories=categories, ignore_tagger=ignore_tagger)

        loss = debias_loss_fn(post_log_probs, post_out_id, post_tok_label_id)
        
        if tagging_loss_fn is not None and ARGS.tag_loss_mixing_prob > 0:
            tok_loss = tagging_loss_fn(tok_logits, pre_tok_label_id, apply_mask=pre_tok_label_id)
            loss = loss + (ARGS.tag_loss_mixing_prob * tok_loss)

        if coverage:
            cov_loss = seq2seq_utils.coverage_loss(attns, coverages)
            loss = loss + ARGS.coverage_lambda * cov_loss


        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        model.zero_grad()
        
        losses.append(loss.detach().cpu().numpy())

    return losses


def run_eval(model, dataloader, tok2id, out_file_path, max_seq_len, beam_width=1):
    id2tok = {x: tok for (tok, x) in tok2id.items()}

    out_file = open(out_file_path, 'w')

    losses = []
    hits = []
    preds, golds, srcs = [], [], []
    for step, batch in enumerate(tqdm(dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch

        post_start_id = tok2id['行']
        max_len = min(max_seq_len, pre_len[0].detach().cpu().numpy() + 10)

        with torch.no_grad():
            predicted_toks, predicted_probs = model.inference_forward(
                pre_id, post_start_id, pre_mask, pre_len, max_len, pre_tok_label_id,
                rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
                beam_width=beam_width)

        new_hits, new_preds, new_golds, new_srcs = seq2seq_utils.dump_outputs(
            pre_id.detach().cpu().numpy(), 
            post_out_id.detach().cpu().numpy(), 
            predicted_toks, 
            pre_tok_label_id.detach().cpu().numpy(), 
            id2tok, out_file,
            pred_dists=predicted_probs)
        hits += new_hits
        preds += new_preds
        golds += new_golds
        srcs += new_srcs
    out_file.close()

    return hits, preds, golds, srcs





