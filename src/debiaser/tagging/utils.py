from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.optimization import BertAdam

import sys; sys.path.append('.')
from shared.args import ARGS
from shared.constants import CUDA


def build_optimizer(model, num_train_steps, learning_rate):
    global ARGS

    if ARGS.tagger_from_debiaser:
        parameters = list(model.cls_classifier.parameters()) + list(
            model.tok_classifier.parameters())
        parameters = list(filter(lambda p: p.requires_grad, parameters))
        return optim.Adam(parameters, lr=ARGS.learning_rate)
    else:
        param_optimizer = list(model.named_parameters())
        param_optimizer = list(filter(lambda name_param: name_param[1].requires_grad, param_optimizer))
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        return BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=0.1,
                             t_total=num_train_steps)


def build_loss_fn(debias_weight=None):
    global ARGS
    
    if debias_weight is None:
        debias_weight = ARGS.debias_weight
    
    weight_mask = torch.ones(ARGS.num_tok_labels)
    weight_mask[-1] = 0

    if CUDA:
        weight_mask = weight_mask.cuda()
        criterion = CrossEntropyLoss(weight=weight_mask).cuda()
        per_tok_criterion = CrossEntropyLoss(weight=weight_mask, reduction='none').cuda()
    else:
        criterion = CrossEntropyLoss(weight=weight_mask)
        per_tok_criterion = CrossEntropyLoss(weight=weight_mask, reduction='none')


    def cross_entropy_loss(logits, labels, apply_mask=None):
        return criterion(
            logits.contiguous().view(-1, ARGS.num_tok_labels), 
            labels.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

    def weighted_cross_entropy_loss(logits, labels, apply_mask=None):
        # weight mask = where to apply weight (post_tok_label_id from the batch)
        weights = apply_mask.contiguous().view(-1)
        weights = ((debias_weight - 1) * weights) + 1.0

        per_tok_losses = per_tok_criterion(
            logits.contiguous().view(-1, ARGS.num_tok_labels), 
            labels.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))
        per_tok_losses = per_tok_losses * weights

        loss = torch.mean(per_tok_losses[torch.nonzero(per_tok_losses)].squeeze())

        return loss

    if debias_weight == 1.0:
        loss_fn = cross_entropy_loss
    else:
        loss_fn = weighted_cross_entropy_loss

    return loss_fn


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def run_inference(model, eval_dataloader, loss_fn, tokenizer):
    global ARGS

    out = {
        'input_toks': [],
        'post_toks': [],

        'tok_loss': [],
        'tok_logits': [],
        'tok_probs': [],
        'tok_labels': [],

        'labeling_hits': []
    }

    for step, batch in enumerate(tqdm(eval_dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)

        ( 
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch

        with torch.no_grad():
            _, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
                rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
                pre_len=pre_len)
            tok_loss = loss_fn(tok_logits, tok_label_id, apply_mask=tok_label_id)
        out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in pre_id.cpu().numpy()]
        out['post_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in post_in_id.cpu().numpy()]
        out['tok_loss'].append(float(tok_loss.cpu().numpy()))
        logits = tok_logits.detach().cpu().numpy()
        labels = tok_label_id.cpu().numpy()
        out['tok_logits'] += logits.tolist()
        out['tok_labels'] += labels.tolist()
        out['tok_probs'] += to_probs(logits, pre_len)
        out['labeling_hits'] += tag_hits(logits, labels)

    return out

def train_for_epoch(model, train_dataloader, loss_fn, optimizer):
    global ARGS
    
    losses = []
    
    for step, batch in enumerate(tqdm(train_dataloader)):
        if ARGS.debug_skip and step > 2:
            continue
    
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        ( 
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch
        _, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
            rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
            pre_len=pre_len)
        loss = loss_fn(tok_logits, tok_label_id, apply_mask=tok_label_id)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        losses.append(loss.detach().cpu().numpy())

    return losses

def to_probs(logits, lens):
    per_tok_probs = softmax(np.array(logits)[:, :, :2], axis=2)
    pos_scores = per_tok_probs[:, :, -1]
    
    out = []
    for score_seq, l in zip(pos_scores, lens):
        out.append(score_seq[:l].tolist())
    return out

def is_ranking_hit(probs, labels, top=1):
    global ARGS
    
    # get rid of padding idx
    [probs, labels] = list(zip(*[(p, l)  for p, l in zip(probs, labels) if l != ARGS.num_tok_labels - 1 ]))
    probs_indices = list(zip(np.array(probs)[:, 1], range(len(labels))))
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    if sum([labels[i] for i in top_indices]) > 0:
        return 1
    else:
        return 0

def tag_hits(logits, tok_labels, top=1):
    global ARGS
    
    probs = softmax(np.array(logits)[:, :, : ARGS.num_tok_labels - 1], axis=2)

    hits = [
        is_ranking_hit(prob_dist, tok_label, top=top) 
        for prob_dist, tok_label in zip(probs, tok_labels)
    ]
    return hits
