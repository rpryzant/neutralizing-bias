import torch
import math
from collections import Counter
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from simplediff import diff

import sys; sys.path.append('.')
from shared.args import ARGS
from shared.constants import CUDA



#############################################################
def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)
#############################################################

def build_loss_fn(vocab_size):
    global ARGS

    weight_mask = torch.ones(vocab_size)
    weight_mask[0] = 0
    criterion = nn.NLLLoss(weight=weight_mask)
    per_tok_criterion = nn.NLLLoss(weight=weight_mask, reduction='none')

    if CUDA:
        weight_mask = weight_mask.cuda()
        criterion = criterion.cuda()
        per_tok_criterion = per_tok_criterion.cuda()

    def cross_entropy_coverage(log_probs, copy_matrices, a_vecs, coverage_vecs, p_gens, labels):
        '''
        For the coverage loss we also need to define a lambda hyper-parameter
        that weighs off between only using cross-entropy loss and minimizing
        the coverage-attention term of the loss.

        The parameters are all passed in as lists which have the length of the
        sequence length of the decoder inputs.
        '''
        lambda_weight = ARGS.lambda_coverage_weight

        p_gens_tensor = torch.stack(p_gens)
        vocab_probs_tensor = torch.stack(log_probs)
        copy_tensor = torch.stack(copy_matrices)

        predictions = p_gens_tensor * vocab_probs_tensor + (1-p_gens_tensor) * copy_tensor
        predictions = predictions.reshape(-1, vocab_size)
        labels = labels.reshape(-1)

        ce_loss = criterion(predictions, labels)

        coverage_loss = 0.0
        for a, c in zip(a_vecs, coverage_vecs):
            min_vec = torch.min(a,c)
            sum = torch.sum(min_vec)
            coverage_loss += sum
        coverage_loss *= lambda_weight
        batch_size = log_probs[0].shape[0]
        seq_len = len(a_vecs)
        coverage_loss /= (batch_size * seq_len)

        return ce_loss + coverage_loss

    def cross_entropy_loss(log_probs, labels, apply_mask=None):
        return criterion(
            log_probs.contiguous().view(-1, vocab_size),
            labels.contiguous().view(-1))

    def weighted_cross_entropy_loss(log_probs, labels, apply_mask=None):
        # weight apply_mask = wehere to apply weight
        weights = apply_mask.contiguous().view(-1)
        weights = ((ARGS.debias_weight - 1) * weights) + 1.0

        per_tok_losses = per_tok_criterion(
            log_probs.contiguous().view(-1, vocab_size),
            labels.contiguous().view(-1))

        per_tok_losses = per_tok_losses * weights

        loss = torch.mean(per_tok_losses[torch.nonzero(per_tok_losses)].squeeze())

        return loss

    if ARGS.debias_weight == 1.0:
        loss_fn = cross_entropy_loss
    elif ARGS:
        loss_fn = weighted_cross_entropy_loss

    if ARGS.pointer_generator:
        loss_fn = cross_entropy_coverage

    return loss_fn, cross_entropy_loss


def train_for_epoch(model, dataloader, tok2id, optimizer, loss_fn, ignore_enrich=False):
    global CUDA
    global ARGS

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
            _, _, _
        ) = batch

        if ARGS.pointer_generator:
            '''
            if the model is a pointer generator we need to return the
            predicted output logits as well as the coverage and attention
            distributions so that these can be fed into the calculation
            of the loss described in: https://arxiv.org/pdf/1704.04368.pdf.
            '''
            (logits_list,
            copy_matrices_list,
            attention_vec_list,
            coverage_vec_list,
            prob_gen_lists) = model(pre_id, post_in_id, pre_mask, pre_len, pre_tok_label_id, ignore_enrich=ignore_enrich)
            # Need post_out_id for CE loss that is added to the additional
            # coverage loss
            loss = loss_fn(logits_list,
                            copy_matrices_list,
                            attention_vec_list,
                            coverage_vec_list,
                            prob_gen_lists,
                            post_out_id)
        else:
            log_probs, probs = model(pre_id, post_in_id, pre_mask, pre_len, pre_tok_label_id, ignore_enrich=ignore_enrich)
            loss = loss_fn(log_probs, post_out_id, post_tok_label_id)

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        model.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        print(loss.item())

    return losses


def dump_outputs(src_ids, gold_ids, predicted_ids, gold_tok_dist, id2tok, out_file,
        pred_dists=None):
    out_hits = []
    preds_for_bleu = []
    golds_for_bleu = []
    srcs_for_bleu = []

    if pred_dists is None:
        pred_dists = [''] * len(src_ids)
    for src_seq, gold_seq, pred_seq, gold_dist, pred_dist in zip(
        src_ids, gold_ids, predicted_ids, gold_tok_dist, pred_dists):

        src_seq = [id2tok[x] for x in src_seq]
        gold_seq = [id2tok[x] for x in gold_seq]
        pred_seq = [id2tok[x] for x in pred_seq[1:]]   # ignore start token
        if '止' in gold_seq:
            gold_seq = gold_seq[:gold_seq.index('止')]
        if '止' in pred_seq:
            pred_seq = pred_seq[:pred_seq.index('止')]

        gold_replace = [chunk for tag, chunk in diff(src_seq, gold_seq) if tag == '+']
        pred_replace = [chunk for tag, chunk in diff(src_seq, pred_seq) if tag == '+']

        src_seq = ' '.join(src_seq).replace('[PAD]', '').strip()
        gold_seq = ' '.join(gold_seq).replace('[PAD]', '').strip()
        pred_seq = ' '.join(pred_seq).replace('[PAD]', '').strip()

        try:
            print('#' * 80, file=out_file)
            print('IN SEQ: \t', src_seq.encode('utf-8'), file=out_file)
            print('GOLD SEQ: \t', gold_seq.encode('utf-8'), file=out_file)
            print('PRED SEQ:\t', pred_seq.encode('utf-8'), file=out_file)
            print('GOLD DIST: \t', list(gold_dist), file=out_file)
            print('PRED DIST: \t', list(pred_dist), file=out_file)
            print('GOLD TOK: \t', list(gold_replace), file=out_file)
            print('PRED TOK: \t', list(pred_replace), file=out_file)
        except UnicodeEncodeError:
            pass

        if gold_seq == pred_seq:
            out_hits.append(1)
        else:
            out_hits.append(0)

        preds_for_bleu.append(pred_seq.split())
        golds_for_bleu.append(gold_seq.split())
        srcs_for_bleu.append(src_seq.split())

    return out_hits, preds_for_bleu, golds_for_bleu, srcs_for_bleu


def run_eval(model, dataloader, tok2id, out_file_path, max_seq_len, beam_width=1):
    global ARGS

    id2tok = {x: tok for (tok, x) in tok2id.items()}

    weight_mask = torch.ones(len(tok2id))
    weight_mask[0] = 0
    criterion = nn.CrossEntropyLoss(weight=weight_mask)

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
            _, _, _
        ) = batch

        post_start_id = tok2id['行']
        max_len = min(max_seq_len, pre_len[0].detach().cpu().numpy() + 10)

        with torch.no_grad():
            predicted_toks = model.inference_forward(
                pre_id, post_start_id, pre_mask, pre_len, max_len, pre_tok_label_id,
                beam_width=beam_width)

        new_hits, new_preds, new_golds, new_srcs = dump_outputs(
            pre_id.detach().cpu().numpy(),
            post_out_id.detach().cpu().numpy(),
            predicted_toks,
            pre_tok_label_id.detach().cpu().numpy(),
            id2tok, out_file)
        hits += new_hits
        preds += new_preds
        golds += new_golds
        srcs += new_srcs
    out_file.close()

    return hits, preds, golds, srcs
