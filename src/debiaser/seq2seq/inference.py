#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python seq2seq.py --test ../../data/v5/final/bias.test --working_dir TEST/ --checkpoint TEST/model_1.ckpt

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

TEST_TEXT = test_data_prefix + '.pre'
TEST_TEXT_POST = test_data_prefix + '.post'

WORKING_DIR = working_dir

NUM_BIAS_LABELS = 2
NUM_TOK_LABELS = 3

if ARGS.bert_encoder:
    TEST_BATCH_SIZE = 16
else:
    TEST_BATCH_SIZE = ARGS.batch_size // ARGS.beam_width

MAX_SEQ_LEN = ARGS.max_seq_len

CUDA = (torch.cuda.device_count() > 0)
                                                                

# # # # # # # # ## # # # ## # # DATA  # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

eval_dataloader, num_eval_examples = get_dataloader(
    TEST_TEXT, TEST_TEXT_POST,
    tok2id, TEST_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok, ARGS=ARGS)




# # # # # # # # ## # # # ## # # MODELS # # # # # # # # ## # # # ## # #
if ARGS.no_tok_enrich:
    model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
else:
    model = seq2seq_model.Seq2SeqEnrich(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('NUM PARAMS: ', params)

if os.path.exists(ARGS.checkpoint):
    print('LOADING FROM ' + ARGS.checkpoint)
    model.load(ARGS.checkpoint)
    print('...DONE')

if CUDA:
    model = model.cuda()


# # # # # # # # # # # # EVAL # # # # # # # # # # # # # #
print('EVAL...')
model.eval()
hits, preds, golds, srcs = utils.run_eval(
    model, eval_dataloader, tok2id, ARGS.checkpoint + '.inference_results.txt',
    MAX_SEQ_LEN, ARGS.beam_width)
print('BLEU:\t\t ' + str(utils.get_bleu(preds, golds)))
print('HITS:\t\t ' + str(np.mean(hits)))




































