# !/usr/bin/env python
# -*- coding: utf-8 -*-

# run minimal job:
# python seq2seq/train.py --train ../../data/v6/corpus.wordbiased.tag.train --test ../../data/v6/corpus.wordbiased.tag.test --working_dir TEST --max_seq_len 15 --train_batch_size 3 --test_batch_size 10  --hidden_size 32 --debug_skip

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

import model as seq2seq_model

import sys; sys.path.append('.')
from shared.args import ARGS
from shared.data import get_dataloader
from shared.constants import CUDA

import utils

BERT_MODEL = "bert-base-uncased"

if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)

with open(ARGS.working_dir + '/command.sh', 'w') as f:
    f.write('python' + ' '.join(sys.argv) + '\n')

if ARGS.bert_encoder:
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
else:
    TRAIN_BATCH_SIZE = ARGS.train_batch_size
    TEST_BATCH_SIZE = ARGS.test_batch_size // ARGS.beam_width


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


if ARGS.pretrain_data:
    pretrain_dataloader, num_pretrain_examples = get_dataloader(
        ARGS.pretrain_data,
        tok2id, TRAIN_BATCH_SIZE, ARGS.working_dir + '/pretrain_data.pkl',
        noise=True)

train_dataloader, num_train_examples = get_dataloader(
    ARGS.train,
    tok2id, TRAIN_BATCH_SIZE, ARGS.working_dir + '/train_data.pkl',
    add_del_tok=ARGS.add_del_tok)
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, TEST_BATCH_SIZE, ARGS.working_dir + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok)


# # # # # # # # ## # # # ## # # MODELS # # # # # # # # ## # # # ## # #
if ARGS.pointer_generator:
    model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), 
        hidden_size=ARGS.hidden_size,
        emb_dim=768, # 768 = bert hidden size
        dropout=0.2, 
        tok2id=tok2id) 
else:
    model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, 
        dropout=0.2, 
        tok2id=tok2id)
if CUDA:
    model = model.cuda()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('NUM PARAMS: ', params)


# # # # # # # # ## # # # ## # # OPTIMIZER, LOSS # # # # # # # # ## # # # ## # #

num_train_steps = (num_train_examples * 40)
if ARGS.pretrain_data: 
    num_train_steps += (num_pretrain_examples * ARGS.pretrain_epochs)

optimizer = utils.build_optimizer(model, num_train_steps)

loss_fn, cross_entropy_loss = utils.build_loss_fn(vocab_size=len(tok2id))

writer = SummaryWriter(ARGS.working_dir)

# # # # # # # # # # # PRETRAINING (optional) # # # # # # # # # # # # # # # #
if ARGS.pretrain_data:
    print('PRETRAINING...')
    for epoch in range(ARGS.pretrain_epochs):
        model.train()
        losses = utils.train_for_epoch(model, pretrain_dataloader, tok2id, optimizer, cross_entropy_loss,
            ignore_enrich=not ARGS.use_pretrain_enrich)
        writer.add_scalar('pretrain/loss', np.mean(losses), epoch)

    print('SAVING DEBIASER...')
    torch.save(model.state_dict(), ARGS.working_dir + '/debiaser.ckpt')

# # # # # # # # # # # # TRAINING # # # # # # # # # # # # # #

for epoch in range(ARGS.epochs):
    print('EPOCH ', epoch)
    print('TRAIN...')
    model.train()
    losses = utils.train_for_epoch(model, train_dataloader, tok2id, optimizer, loss_fn, coverage=ARGS.coverage)
    writer.add_scalar('train/loss', np.mean(losses), epoch+1)

    print('SAVING...')
    model.save(ARGS.working_dir + '/model_%d.ckpt' % (epoch+1))

    print('EVAL...')
    model.eval()
    hits, preds, golds, srcs = utils.run_eval(
        model, eval_dataloader, tok2id, ARGS.working_dir + '/results_%d.txt' % epoch,
        ARGS.max_seq_len, ARGS.beam_width)
    # writer.add_scalar('eval/partial_bleu', utils.get_partial_bleu(preds, golds, srcs), epoch+1)
    writer.add_scalar('eval/bleu', utils.get_bleu(preds, golds), epoch+1)
    writer.add_scalar('eval/true_hits', np.mean(hits), epoch+1)
