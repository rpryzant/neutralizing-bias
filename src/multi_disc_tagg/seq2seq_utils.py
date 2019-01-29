import torch
import math
from collections import Counter
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from simplediff import diff


CUDA = (torch.cuda.device_count() > 0)


########################################################
####### PARTIAL BLEU....TODO THINK ABOUT THIS ##########
########################################################

def partial_bleu_stats(hypothesis, reference, source):
    """Compute statistics for BLEU."""
    ref_unique = set(reference) - set(source) 
    # print(ref_unique)
    # print(diff(reference, source))

    def valid_ngram(ngram):
        return len(set(ngram) & ref_unique) > 0

    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)
            if valid_ngram(reference[i:i + n])]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))

    return stats


def partial_bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    # print(c, r, stats)
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_partial_bleu(hypotheses, reference, source):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref, src in zip(hypotheses, reference, source):
        stats += np.array(partial_bleu_stats(hyp, ref, src))
    print(100 * partial_bleu(stats))
    return 100 * partial_bleu(stats)




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


def train_for_epoch(model, dataloader, tok2id, optimizer, loss_fn, ignore_enrich=False):
    global CUDA

    losses = []
    for step, batch in enumerate(tqdm(dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, post_tok_label_id, tok_dist,
            replace_id, _, _, type_id
        ) = batch

        post_logits, post_probs = model(pre_id, post_in_id, pre_mask, pre_len, tok_dist, type_id, ignore_enrich=ignore_enrich)
        loss = loss_fn(post_logits, post_out_id, post_tok_label_id)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        model.zero_grad()

        losses.append(loss.detach().cpu().numpy())

    return losses


def dump_outputs(src_ids, gold_ids, predicted_ids, gold_replace_id, gold_tok_dist, id2tok, out_file):
    out_hits = []
    preds_for_bleu = []
    golds_for_bleu = []
    srcs_for_bleu = []

    for src_seq, gold_seq, pred_seq, gold_replace, gold_dist in zip(
        src_ids, gold_ids, predicted_ids, gold_replace_id, gold_tok_dist):

        src_seq = [id2tok[x] for x in src_seq]
        gold_seq = [id2tok[x] for x in gold_seq]
        pred_seq = [id2tok[x] for x in pred_seq[1:]]   # ignore start token
        if '止' in gold_seq:
            gold_seq = gold_seq[:gold_seq.index('止')]
        if '止' in pred_seq:
            pred_seq = pred_seq[:pred_seq.index('止')]
        src_seq = ' '.join(src_seq).replace('[PAD]', '').strip()
        gold_seq = ' '.join(gold_seq).replace('[PAD]', '').strip()
        pred_seq = ' '.join(pred_seq).replace('[PAD]', '').strip()

        gold_replace = id2tok[gold_replace]
        pred_replace = [chunk for tag, chunk in diff(src_seq.split(), pred_seq.split()) if tag == '+']
        try:
            print('#' * 80, file=out_file)
            print('IN SEQ: \t', src_seq.encode('utf-8'), file=out_file)
            print('GOLD SEQ: \t', gold_seq.encode('utf-8'), file=out_file)
            print('PRED SEQ:\t', pred_seq.encode('utf-8'), file=out_file)
            print('GOLD DIST: \t', list(gold_dist), file=out_file)
            print('GOLD TOK: \t', gold_replace.encode('utf-8'), file=out_file)
            print('PRED TOK: \t', pred_replace, file=out_file)
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
    id2tok = {x: tok for (tok, x) in tok2id.items()}

    weight_mask = torch.ones(len(tok2id))
    weight_mask[0] = 0
    criterion = nn.CrossEntropyLoss(weight=weight_mask)

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
            replace_id, _, _, type_id, _
        ) = batch

        post_start_id = tok2id['行']
        max_len = min(max_seq_len, pre_len[0].detach().cpu().numpy() + 10)

        with torch.no_grad():
            predicted_toks = model.inference_forward(
                pre_id, post_start_id, pre_mask, pre_len, max_len, tok_dist, type_id,
                beam_width=beam_width)

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
