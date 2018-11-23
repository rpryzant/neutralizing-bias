import math
import numpy as np
import sys
from collections import Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
import editdistance


import data
from cuda import CUDA

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BLEU functions from https://github.com/MaximumEntropy/Seq2Seq-PyTorch
#    (ran some comparisons, and it matches moses's multi-bleu.perl)
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_edit_distance(hypotheses, reference):
    ed = 0
    for hyp, ref in zip(hypotheses, reference):
        ed += editdistance.eval(hyp, ref)

    return ed * 1.0 / len(hypotheses)


def decode_minibatch(max_len, start_id, model, src_input, srclens, srcmask):
    """ argmax decoding """
    # Initialize target with <s> for every sentence
    tgt_input = Variable(torch.LongTensor(
        [
            [start_id] for i in range(src_input.size(0))
        ]
    ))
    if CUDA:
        tgt_input = tgt_input.cuda()

    for i in range(max_len):
        # run input through the model
        decoder_logit, word_probs = model(src_input, tgt_input, srcmask, srclens)
        decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
        # select the predicted "next" tokens, attach to target-side inputs
        next_preds = Variable(torch.from_numpy(decoder_argmax[:, -1]))
        if CUDA:
            next_preds = next_preds.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

    return tgt_input

def decode_dataset(model, src, tgt, config):
    """Evaluate model."""
    preds = []
    ground_truths = []
    for j in range(0, len(src['data']), config['data']['batch_size']):
        sys.stdout.write("\r%s/%s..." % (j, len(src['data'])))
        sys.stdout.flush()

        # get batch
        src_input, src_output, srclens, srcmask, idx = data.get_minibatch(
            src['data'], src['tok2id'],
            j, config['data']['batch_size'],
            config['data']['max_len'], sort=True)

        _, tgt_output, _, _, _ = data.get_minibatch(
                tgt['data'], tgt['tok2id'],
                j, config['data']['batch_size'],
                config['data']['max_len'], idx=idx)

        # TODO -- beam search
        tgt_pred = decode_minibatch(
            config['data']['max_len'], tgt['tok2id']['<s>'], 
            model, src_input, srclens, srcmask)

        # convert seqs to tokens
        tgt_pred = tgt_pred.data.cpu().numpy()
        tgt_pred = [
            [tgt['id2tok'][x] for x in line]
            for line in tgt_pred]
        tgt_output = tgt_output.data.cpu().numpy()
        tgt_output = [
            [tgt['id2tok'][x] for x in line]
            for line in tgt_output]

        tgt_pred = data.unsort(tgt_pred, idx)
        tgt_output = data.unsort(tgt_output, idx)

        # cut off at </s>
        for tgt_pred, tgt_gold in zip(tgt_pred, tgt_output):
            idx = tgt_pred.index('</s>') if '</s>' in tgt_pred else len(tgt_pred)
            preds.append(tgt_pred[:idx + 1])

            idx = tgt_gold.index('</s>') if '</s>' in tgt_gold else len(tgt_gold)
            # append start to golds only, because preds were kickstarted w/them
            ground_truths.append(['<s>'] + tgt_gold[:idx + 1])

    return preds, ground_truths


def inference_metrics(model, src, tgt, config):
    """ decode and evaluate bleu """
    preds, ground_truths = decode_dataset(
        model, src, tgt, config)
    bleu = get_bleu(preds, ground_truths)
    edit_distance = get_edit_distance(preds, ground_truths)
    preds = [' '.join(seq) for seq in preds]
    ground_truths = [' '.join(seq) for seq in ground_truths]

    return bleu, edit_distance, preds, ground_truths


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
        src_input, src_output, srclens, srcmask, idx = data.get_minibatch(
            src['data'], src['tok2id'],
            j, config['data']['batch_size'],
            config['data']['max_len'], sort=True)

        tgt_input, tgt_output, _, _, _ = data.get_minibatch(
                tgt['data'], tgt['tok2id'],
                j, config['data']['batch_size'],
                config['data']['max_len'], idx=idx)

        decoder_logit, decoder_probs = model(src_input, tgt_input, srcmask, srclens)

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, len(tgt['tok2id'])),
            tgt_output.view(-1)
        )
        losses.append(loss.data[0])

    return np.mean(losses)
