import math
import numpy as np
import sys
from collections import Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
import editdistance

import models
import data
import ops

from cuda import CUDA




def get_precisions_recalls_DEPRECIATED(inputs, preds, ground_truths):
    """ v1 of precision/recall based on some dumb logic """
    def precision_recall(src, tgt, pred):
        """
        src: [string tokens], the input to the model
        tgt: [string tokens], the gold targets
        pred: [string tokens], the model outputs
        """
        tgt_unique = set(tgt) - set(src)
        src_unique = set(src) - set(tgt)
        
        # new words the model correctly introduced
        true_positives = len(set(pred) & tgt_unique)
        # new words the model incorrectly introduced
        false_positives = len(set(pred) - set(src) - set(tgt))
        # old words the model incorrectly retained
        false_negatives = len(set(pred) & src_unique)
        
        precision = true_positives * 1.0 / (true_positives + false_positives + 0.001)
        recall = true_postitives * 1.0 / (true_positives + false_negatives + 0.001)

        return precision, recall

    [precisions, recalls] = list(zip(*[
        precision_recall(src, tgt, pred) 
        for src, tgt, pred in zip(inputs, ground_truths, preds)
    ]))

    return precisions, recalls


#########################################################################
# ABOVE THIS LINE ARE DEPRECIATED METHODS...TREAD CAREFULLY
#########################################################################




def bleu_stats(hypothesis, reference, word_list=None):
    """Compute statistics for BLEU."""

    def is_valid_ngram(ngram):
        if word_list is None:
            return True
        else:
            return len(set(word_list) & set(ngram)) > 0

    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter([
            tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)
            if is_valid_ngram(hypothesis[i:i + n])
        ])
        r_ngrams = Counter([
            tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)
            if is_valid_ngram(reference[i:i + n])
        ])
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

def get_bleu(hypotheses, reference, word_lists=None):
    """Get validation BLEU score for dev set.
        If provided with a list of word lists, then we'll only consider
            ngrams with words from that list.
    """
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    if word_lists is None:
        word_lists = [None for _ in range(len(hypotheses))]

    for hyp, ref, wlist in zip(hypotheses, reference, word_lists):
        stats += np.array(bleu_stats(hyp, ref, word_list=wlist))
    return 100 * bleu(stats)


def get_precision_recall(inputs, top_k_preds, ground_truths, k=None):
    """
    Precision@k = (# of generated candidates @k that are relevant to targets) / (# of generated candidates @k)
    Recall@k = (# of generated candidates @k that are relevant to targets) / (total # of relevant targets)
    
    top_k_preds: [Batch, length, k]
    """
    if not k:
        k = len(top_k_preds[0][0])
    
    def precision_recall(src, tgt, top_k):
        tgt_unique = set(tgt) - set(src)
        pred_toks = [tok for klist in top_k for tok in klist[:k]]
        precision = len(tgt_unique & set(pred_toks)) * 1.0 / (len(pred_toks) + 0.0001)
        recall = len(tgt_unique & set(pred_toks)) * 1.0 / (len(tgt_unique) + 0.0001)

        return precision, recall

    [precisions, recalls] = list(zip(*[
        precision_recall(src, tgt, pred) 
        for src, tgt, pred in zip(inputs, ground_truths, top_k_preds)
    ]))

    return np.average(precisions), np.average(recalls)



def get_edit_distance(hypotheses, reference):
    ed = 0
    for hyp, ref in zip(hypotheses, reference):
        ed += editdistance.eval(hyp, ref)

    return ed * 1.0 / len(hypotheses)


def decode_minibatch(max_len, start_id, model, src_input, srclens, srcmask,
        aux_input, auxlens, auxmask, side_info, k):
    """ argmax decoding """
    # Initialize target with <s> for every sentence
    tgt_input = Variable(torch.LongTensor([
            [start_id] for i in range(src_input.size(0))
    ]))
    if CUDA:
        tgt_input = tgt_input.cuda()

    top_k_toks = []
    for i in range(max_len):
        # run input through the model
        decoder_logit, word_probs, _, _ = model(src_input, tgt_input, srcmask, srclens,
            aux_input, auxmask, auxlens, side_info)

        # logits for the latest timestep
        word_probs = word_probs.data.cpu().numpy()[:, -1, :]
        # sorted indices (descending)
        sorted_indices = np.argsort((word_probs))[:, ::-1]
        # select the predicted "next" tokens, attach to target-side inputs
        next_preds = Variable(torch.from_numpy(sorted_indices[:, 0]))
        if CUDA:
            next_preds = next_preds.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)
        # remember the top k indices at this step for evaluation
        top_k_toks.append( sorted_indices[:, :k] )

    # make top_k_toks into [Batch, Length, k] tensor
    top_k_toks = np.array(top_k_toks)
    top_k_toks = np.transpose(top_k_toks, (1, 0, 2))

    # make sure the top k=1 tokens is equal to the true model predictions (argmax)
    assert np.array_equal(
        top_k_toks[:, :, 0], 
        tgt_input[:, 1:].data.cpu().numpy()) # ignore <s> kickstart

    return top_k_toks

# convert seqs to tokens
def ids_to_toks(tok_seqs, id2tok, sort_indices, cuts=None, save_cuts=False):
    out = []
    cut_indices = []
    # take off the gpu
    if isinstance(tok_seqs, torch.Tensor):
        tok_seqs = tok_seqs.cpu().numpy()
    # convert to toks, cut off at </s>
    for i, line in enumerate(tok_seqs):
        toks = [id2tok[x] for x in line]
        if cuts is not None:
            cut_idx = cuts[i]
        elif '</s>' in toks:
            cut_idx = toks.index('</s>')
        else:
            cut_idx = len(toks)
        out.append( toks[:cut_idx] )
        cut_indices += [cut_idx]
    # unsort
    out = data.unsort(out, sort_indices)

    if save_cuts:
        return out, cut_indices
    else:
        return out

def decode_dataset(model, src, tgt, config, k=20):
    """Evaluate model."""
    inputs = []
    preds = []
    top_k_preds = []
    auxs = []
    ground_truths = []
    raw_srcs = []
    for j in range(0, len(src['data']), config['data']['batch_size']):
        sys.stdout.write("\r%s/%s..." % (j, len(src['data'])))
        sys.stdout.flush()

        # get batch
        input_content, input_aux, output, side_info, raw_src = data.minibatch(
            src, tgt, j, 
            config['data']['batch_size'], 
            config['data']['max_len'], 
            config,
            is_test=True)
        input_lines_src, output_lines_src, srclens, srcmask, indices = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output
        _, raw_src, _, _, _ = raw_src
        side_info, _, _, _, _ = side_info

        # TODO -- beam search
        tgt_pred_top_k = decode_minibatch(
            config['data']['max_len'], tgt['tok2id']['<s>'], 
            model, input_lines_src, srclens, srcmask,
            input_ids_aux, auxlens, auxmask, side_info, k=k)

        # convert inputs/preds/targets/aux to human-readable form
        inputs += ids_to_toks(output_lines_src, src['id2tok'], indices)
        ground_truths += ids_to_toks(output_lines_tgt, tgt['id2tok'], indices)
        raw_srcs += ids_to_toks(raw_src, src['id2tok'], indices)

        # TODO -- refactor this stuff!! it's shitty
        # get the "offical" predictions from the model
        pred_toks, pred_lens = ids_to_toks(
            tgt_pred_top_k[:, :, 0], tgt['id2tok'], indices, save_cuts=True)
        preds += pred_toks
        # now get all the other top-k prediction levels
        top_k_pred = [pred_toks]
        for i in range(k - 1):
            top_k_pred.append(ids_to_toks(
                tgt_pred_top_k[:, :, i + 1], tgt['id2tok'], indices, cuts=pred_lens)
            )
        # top_k_pred is [k, batch, length] where length is ragged
        # but we want it in [batch, length, k]. Manual transpose b/c ragged :( 
        batch_size = len(top_k_pred[0]) # could be variable at test time
        pred_lens = data.unsort(pred_lens, indices)
        top_k_pred_transposed = [[] for _ in range(batch_size)]
        for bi in range(batch_size):
            for ti in range(pred_lens[bi]):
                top_k_pred_transposed[bi] += [[
                    top_k_pred[ki][bi][ti] for ki in range(k)
                ]]
        top_k_preds += top_k_pred_transposed
        
        if config['model']['model_type'] == 'delete':
            auxs += [[str(x)] for x in input_ids_aux.data.cpu().numpy()] # because of list comp in inference_metrics()
        elif config['model']['model_type'] == 'delete_retrieve':
            auxs += ids_to_toks(input_ids_aux, tgt['id2tok'], indices)
        elif config['model']['model_type'] == 'seq2seq':
            auxs += ['None' for _ in range(batch_size)]

    return inputs, preds, top_k_preds, ground_truths, auxs, raw_srcs


def get_metrics(inputs, preds, ground_truths, top_k_preds=None, classifier=None):
    bleu = get_bleu(preds, ground_truths)
    src_bleu = get_bleu(
        preds, inputs,
        word_lists=[
            set(src) - set(tgt) for src, tgt in zip(inputs, ground_truths)
        ]
    )
    tgt_bleu = get_bleu(
        preds, ground_truths,
        word_lists=[
            set(tgt) - set(src) for src, tgt in zip(inputs, ground_truths)
        ]
    )

    edit_distance = get_edit_distance(preds, ground_truths)

    if top_k_preds is None:
        top_k_preds = [[[x] for x in seq] for seq in preds]
    tgt_precision, tgt_recall = get_precision_recall(inputs, top_k_preds, ground_truths)
    src_precision, src_recall = get_precision_recall(ground_truths, top_k_preds, inputs)

    if classifier is not None:
        classifier_error = classifier.error_rate(
            seqs=[' '.join(seq) for seq in preds],
            Y=[1 for _ in range(len(preds))])   # we're trying to create "target" seqs
    else:
        classifier_error = -1

    return {
        'bleu':             bleu,
        'src_bleu':         src_bleu,
        'tgt_bleu':         tgt_bleu,
        'edit_distance':    edit_distance,
        'tgt_precision':    tgt_precision,
        'src_precision':    src_precision,
        'tgt_recall':       tgt_recall,
        'src_recall':       src_recall,
        'classifier_error': classifier_error
    }


def inference_metrics(model, src, tgt, config):
    """ decode and evaluate bleu """
    inputs, preds, top_k_preds, ground_truths, auxs, raw_srcs = decode_dataset(
        model, src, tgt, config, k=config['eval']['precision_recall_k'])

    eval_classifier = models.TextClassifier.from_pickle(
        config['eval']['classifier_path'])

    metrics = get_metrics(
        raw_srcs, preds, ground_truths, 
        top_k_preds=top_k_preds, classifier=eval_classifier)

    inputs = [' '.join(seq) for seq in inputs]
    preds = [' '.join(seq) for seq in preds]
    ground_truths = [' '.join(seq) for seq in ground_truths]
    auxs = [' '.join(seq) for seq in auxs]

    return metrics, inputs, preds, ground_truths, auxs


def evaluate_lpp(model, src, tgt, config):
    """ evaluate log perplexity WITHOUT decoding
        (i.e., with teacher forcing)
    """
    weight_mask = torch.ones(len(tgt['tok2id']))
    if CUDA:
        weight_mask = weight_mask.cuda()
    weight_mask[tgt['tok2id']['<pad>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
    if CUDA:
        loss_criterion = loss_criterion.cuda()

    losses = []
    for j in range(0, len(src['data']), config['data']['batch_size']):
        sys.stdout.write("\r%s/%s..." % (j, len(src['data'])))
        sys.stdout.flush()

        # get batch
        input_content, input_aux, output, side_info, _ = data.minibatch(
            src, tgt, j, 
            config['data']['batch_size'], 
            config['data']['max_len'], 
            config,
            is_test=True)
        input_lines_src, _, srclens, srcmask, _ = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output
        side_info, _, _, _, _ = side_info

        decoder_logit, decoder_probs, _, _ = model(
            input_lines_src, input_lines_tgt, srcmask, srclens,
            input_ids_aux, auxlens, auxmask,
            side_info)

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, len(tgt['tok2id'])),
            output_lines_tgt.view(-1)
        )
        losses.append(loss.data[0])

    return np.mean(losses)
