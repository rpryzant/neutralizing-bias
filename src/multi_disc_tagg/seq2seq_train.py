#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python seq2seq.py --train ../../data/v5/final/bias --test ../../data/v5/final/bias --working_dir TEST/

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import os
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from simplediff import diff
import pickle
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import Counter
import math
import functools

from pytorch_pretrained_bert.modeling import BertEmbeddings
from pytorch_pretrained_bert.optimization import BertAdam

from data import get_dataloader
import seq2seq_model
from seq2seq_args import ARGS

import seq2seq_utils as utils


BERT_MODEL = "bert-base-uncased"

# TODO REFACTER AWAY ALL THIS JUNK

train_data_prefix = ARGS.train
test_data_prefix = ARGS.test

working_dir = ARGS.working_dir
if not os.path.exists(working_dir):
    os.makedirs(working_dir)


TRAIN_TEXT = train_data_prefix + '.train.pre'
TRAIN_TEXT_POST = train_data_prefix + '.train.post'

TEST_TEXT = test_data_prefix + '.test.pre'
TEST_TEXT_POST = test_data_prefix + '.test.post'

WORKING_DIR = working_dir

NUM_BIAS_LABELS = 2
NUM_TOK_LABELS = 3


if ARGS.bert_encoder:
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
else:
    TRAIN_BATCH_SIZE = ARGS.batch_size
    TEST_BATCH_SIZE = ARGS.batch_size // ARGS.beam_width

EPOCHS = ARGS.epochs

MAX_SEQ_LEN = ARGS.max_seq_len

CUDA = (torch.cuda.device_count() > 0)
                                                                


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


if ARGS.pretrain_data: 
    pretrain_dataloader, num_pretrain_examples = get_dataloader(
        ARGS.pretrain_data, ARGS.pretrain_data, 
        tok2id, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/pretrain_data.pkl',
        noise=True)

train_dataloader, num_train_examples = get_dataloader(
    TRAIN_TEXT, TRAIN_TEXT_POST, 
    tok2id, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/train_data.pkl',
    add_del_tok=ARGS.add_del_tok)
eval_dataloader, num_eval_examples = get_dataloader(
    TEST_TEXT, TEST_TEXT_POST,
    tok2id, TEST_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok)



# # # # # # # # ## # # # ## # # MODELS # # # # # # # # ## # # # ## # #
if ARGS.no_tok_enrich:
    model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
else:
    model = seq2seq_model.Seq2SeqEnrich(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
if CUDA:
    model = model.cuda()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('NUM PARAMS: ', params)



# # # # # # # # ## # # # ## # # OPTIMIZER # # # # # # # # ## # # # ## # #
writer = SummaryWriter(WORKING_DIR)

if ARGS.bert_encoder:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    num_train_steps = (num_train_examples * 40)
    if ARGS.pretrain_data: 
        num_train_steps += (num_pretrain_examples * ARGS.pretrain_epochs)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-5,
                         warmup=0.1,
                         t_total=num_train_steps)

else:
    optimizer = optim.Adam(model.parameters(), lr=0.0003)


# # # # # # # # ## # # # ## # # LOSS # # # # # # # # ## # # # ## # #
# TODO -- REFACTOR THIS BIG TIME!

weight_mask = torch.ones(len(tok2id))
weight_mask[0] = 0
criterion = nn.CrossEntropyLoss(weight=weight_mask)
per_tok_criterion = nn.CrossEntropyLoss(weight=weight_mask, reduction='none')

if CUDA:
    weight_mask = weight_mask.cuda()
    criterion = criterion.cuda()
    per_tok_criterion = per_tok_criterion.cuda()

def cross_entropy_loss(logits, labels, weight_mask=None):
    return criterion(
        logits.contiguous().view(-1, len(tok2id)), 
        labels.contiguous().view(-1))


def weighted_cross_entropy_loss(logits, labels, weight_mask=None):
    # weight mask = wehere to apply weight
    weights = weight_mask.contiguous().view(-1)
    weights = ((ARGS.debias_weight - 1) * weights) + 1.0

    per_tok_losses = per_tok_criterion(
        logits.contiguous().view(-1, len(tok2id)), 
        labels.contiguous().view(-1))

    per_tok_losses = per_tok_losses * weights

    loss = torch.mean(per_tok_losses[torch.nonzero(per_tok_losses)].squeeze())

    return loss

if ARGS.debias_weight == 1.0:
    loss_fn = cross_entropy_loss
else:
    loss_fn = weighted_cross_entropy_loss

# # # # # # # # # # # PRETRAINING (optional) # # # # # # # # # # # # # # # #
if ARGS.pretrain_data:
    print('PRETRAINING...')
    for epoch in range(ARGS.pretrain_epochs):
        model.train()
        losses = utils.train_for_epoch(model, pretrain_dataloader, tok2id, optimizer, cross_entropy_loss)
        writer.add_scalar('pretrain/loss', np.mean(losses), epoch)



# # # # # # # # # # # # TRAINING # # # # # # # # # # # # # #
print('INITIAL EVAL...')
# model.eval()
# hits, preds, golds = utils.run_eval(
#     model, eval_dataloader, tok2id, WORKING_DIR + '/results_initial.txt',
#     MAX_SEQ_LEN, ARGS.beam_width)
# writer.add_scalar('eval/bleu', utils.get_bleu(preds, golds), 0)
# writer.add_scalar('eval/true_hits', np.mean(hits), 0)

for epoch in range(EPOCHS):
    print('EPOCH ', epoch)
    print('TRAIN...')
    model.train()
    losses = utils.train_for_epoch(model, train_dataloader, tok2id, optimizer, loss_fn)
    writer.add_scalar('train/loss', np.mean(losses), epoch+1)
    
    print('SAVING...')
    model.save(WORKING_DIR + '/model_%d.ckpt' % (epoch+1))

    print('EVAL...')
    model.eval()
    hits, preds, golds = utils.run_eval(
        model, eval_dataloader, tok2id, WORKING_DIR + '/results_%d.txt' % epoch,
        MAX_SEQ_LEN, ARGS.beam_width)
    writer.add_scalar('eval/bleu', utils.get_bleu(preds, golds), epoch+1)
    writer.add_scalar('eval/true_hits', np.mean(hits), epoch+1)






































