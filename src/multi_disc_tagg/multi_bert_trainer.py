# -*- coding: utf-8 -*-
"""
train bert 

python multi_bert_trainer.py --train ../../data/v4/tok/biased --test ../../data/v4/tok/biased --working_dir TEST/
"""
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
import torch
import pickle
import sys
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
import argparse
import sklearn.metrics as metrics

import modeling
from data import get_dataloader

from tagging_args import ARGS

train_data_prefix = ARGS.train
test_data_prefix = ARGS.test
mode = 'tagging_always'#sys.argv[3]
working_dir = ARGS.working_dir
if not os.path.exists(working_dir):
    os.makedirs(working_dir)


assert mode in ['multi_always', 'multi_targeted', 'classification', 'tagging_always', 'tagging_targeted']


TRAIN_TEXT = train_data_prefix + '.train.pre'
TRAIN_TEXT_POST = train_data_prefix + '.train.post'

TEST_TEXT = test_data_prefix + '.test.pre'
TEST_TEXT_POST = test_data_prefix + '.test.post'

WORKING_DIR = working_dir

NUM_BIAS_LABELS = 2
NUM_TOK_LABELS = 3

BERT_MODEL = "bert-base-uncased"

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 5e-5

MAX_SEQ_LEN = 80

CUDA = (torch.cuda.device_count() > 0)

if CUDA:
    print('USING CUDA')



def make_optimizer(model, num_train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=LEARNING_RATE,
                         warmup=0.1,
                         t_total=num_train_steps)
    return optimizer

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def run_inference(model, eval_dataloader, cls_criterion, tok_criterion):
    out = {
        'input_toks': [],

        'tok_loss': [],
        'tok_logits': [],
        'tok_labels': [],

        'labeling_hits': []
    }

    for step, batch in enumerate(tqdm(eval_dataloader)):
        # if step > 1:
        #     continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)

        pre_id, pre_mask, pre_len, post_in_id, post_out_id, tok_label_id, replace_id, rel_ids, pos_ids = batch

        with torch.no_grad():
            bias_logits, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
                rel_ids=rel_ids, pos_ids=pos_ids)

            tok_loss = tok_criterion(
                tok_logits.contiguous().view(-1, NUM_TOK_LABELS), 
                tok_label_id.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))
            
        out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in pre_id.cpu().numpy()]

        out['tok_loss'].append(float(tok_loss.cpu().numpy()))
        logits = tok_logits.detach().cpu().numpy()
        labels = tok_label_id.cpu().numpy()
        out['tok_logits'] += logits.tolist()
        out['tok_labels'] += labels.tolist()
        out['labeling_hits'] += tag_hits(logits, labels)

    return out

def classification_accuracy(logits, labels):
    probs = softmax(np.array(logits), axis=1)
    preds = np.argmax(probs, axis=1)
    return metrics.accuracy_score(labels, preds)



def is_ranking_hit(probs, labels, top=1):
    # get rid of padding idx
    [probs, labels] = list(zip(*[(p, l)  for p, l in zip(probs, labels) if l != NUM_TOK_LABELS - 1 ]))
    probs_indices = list(zip(np.array(probs)[:, 1], range(len(labels))))
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    if sum([labels[i] for i in top_indices]) > 0:
        return 1
    else:
        return 0

def tag_hits(logits, tok_labels, top=1):
    probs = softmax(np.array(logits)[:, :, : NUM_TOK_LABELS - 1], axis=2)

    hits = [
        is_ranking_hit(prob_dist, tok_label, top=top) 
        for prob_dist, tok_label in zip(probs, tok_labels)
    ]
    return hits


print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

train_dataloader, num_train_examples = get_dataloader(
    TRAIN_TEXT, TRAIN_TEXT_POST, 
    tok2id, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/train_data.pkl')
eval_dataloader, num_eval_examples = get_dataloader(
    TEST_TEXT, TEST_TEXT_POST,
    tok2id, TEST_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/test_data.pkl',
    test=True)



print('BUILDING MODEL...')
if ARGS.extra_features_top:
    model = modeling.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            BERT_MODEL,
            cls_num_labels=NUM_BIAS_LABELS,
            tok_num_labels=NUM_TOK_LABELS,
            cache_dir=WORKING_DIR + '/cache',
            tok2id=tok2id,
            args=ARGS)
elif ARGS.extra_features_bottom:
    model = modeling.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            BERT_MODEL,
            cls_num_labels=NUM_BIAS_LABELS,
            tok_num_labels=NUM_TOK_LABELS,
            cache_dir=WORKING_DIR + '/cache',
            tok2id=tok2id,
            args=ARGS)
else:
    model = modeling.BertForMultitask.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        cache_dir=WORKING_DIR + '/cache',
        tok2id=tok2id)
if CUDA:
    model = model.cuda()

print('PREPPING RUN...')
optimizer = make_optimizer(model, int((num_train_examples * EPOCHS) / TRAIN_BATCH_SIZE))

weight_mask = torch.ones(NUM_TOK_LABELS)
weight_mask[-1] = 0

if CUDA:
    weight_mask = weight_mask.cuda()
    tok_criterion = CrossEntropyLoss(weight=weight_mask).cuda()
    cls_criterion = CrossEntropyLoss()
else:
    tok_criterion = CrossEntropyLoss(weight=weight_mask)
    cls_criterion = CrossEntropyLoss()



writer = SummaryWriter(WORKING_DIR)


print('INITIAL EVAL...')
model.eval()
results = run_inference(model, eval_dataloader, cls_criterion, tok_criterion)
writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), 0)

print('TRAINING...')
model.train()
train_step = 0
for epoch in range(EPOCHS):
    print('STARTING EPOCH ', epoch)
    for step, batch in enumerate(tqdm(train_dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        pre_id, pre_mask, pre_len, post_in_id, post_out_id, tok_label_id, replace_id, rel_ids, pos_ids = batch
        bias_logits, tok_logits = model(pre_id, attention_mask=1.0-pre_mask, 
            rel_ids=rel_ids, pos_ids=pos_ids)
        loss = tok_criterion(
            tok_logits.contiguous().view(-1, NUM_TOK_LABELS), 
            tok_label_id.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

        # # only backprop on biased examples if so
        # tok_losses = []
        # for tok_log, tok_lab, bias_lab in zip(tok_logits, tok_label_ids, bias_label_ids):
        #     if mode in ['multi_targeted', 'tagging_targeted'] and int(bias_lab.detach().cpu().numpy()) == 0:
        #         continue
        #     tok_losses.append(tok_criterion(tok_log.view(-1, NUM_TOK_LABELS), tok_lab.view(-1)))
        # if not tok_losses:
        #     tok_loss = 0
        # else:
        #     tok_loss = sum(tok_losses) / len(tok_losses)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        writer.add_scalar('train/loss', loss.data[0], train_step)
        train_step += 1

    # eval
    print('EVAL...')
    model.eval()
    results = run_inference(model, eval_dataloader, cls_criterion, tok_criterion)
    writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), epoch + 1)
    writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), epoch + 1)

    model.train()

