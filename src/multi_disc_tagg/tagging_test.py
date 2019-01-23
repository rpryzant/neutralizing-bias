# -*- coding: utf-8 -*-
"""
train bert 

python tagging_train.py --train ../../data/v5/final/bias --test ../../data/v5/final/bias --working_dir TEST/
"""
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle
import sys
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
import argparse
import sklearn.metrics as metrics
import codecs

import tagging_model
from seq2seq_data import get_dataloader

from tagging_args import ARGS
import tagging_utils





train_data_prefix = ARGS.train
test_data_prefix = ARGS.test
if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)

TEST_TEXT = test_data_prefix + '.pre'
TEST_TEXT_POST = test_data_prefix + '.post'

CUDA = (torch.cuda.device_count() > 0)

if CUDA:
    print('USING CUDA')



print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

eval_dataloader, num_eval_examples = get_dataloader(
    TEST_TEXT, TEST_TEXT_POST,
    tok2id, ARGS.test_batch_size, ARGS.max_seq_len, ARGS.working_dir + '/decoding_data.pkl',
    test=True, ARGS=ARGS, sort_batch=False)


print('BUILDING MODEL...')
if ARGS.extra_features_top:
    model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_bias_labels,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id,
            args=ARGS)
elif ARGS.extra_features_bottom:
    model = tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_bias_labels,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id,
            args=ARGS)
else:
    model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_bias_labels,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache',
        tok2id=tok2id)
        
if os.path.exists(ARGS.checkpoint):
    print('LOADING FROM ' + ARGS.checkpoint)
    model.load_state_dict(torch.load(ARGS.checkpoint))
    print('...DONE')
               
if CUDA:
    model = model.cuda()

print('PREPPING RUN...')

loss_fn = tagging_utils.build_loss_fn(ARGS)
writer = SummaryWriter(ARGS.working_dir)


print('INITIAL EVAL...')
model.eval()
results = tagging_utils.run_inference(model, eval_dataloader, loss_fn, tokenizer)
print('LOSS:\t\t ' + str(np.mean(results['tok_loss'])))
print('ACC:\t\t ' + str(np.mean(results['labeling_hits'])))

with open(ARGS.out_prefix + '.probs', 'w') as f:
    for seq in results['tok_probs']:
        f.write(' '.join([str(x) for x in seq]) + '\n')
# re-write pre/post data because of skipping
with codecs.open(ARGS.out_prefix + '.pre', 'w', 'utf-8') as f:
    for seq in results['input_toks']:
        f.write(' '.join(seq).replace('[PAD]', '').strip() + '\n')  # rm pads so everything lines up

with codecs.open(ARGS.out_prefix + '.post', 'w', 'utf-8') as f:
    for seq in results['post_toks']:
        f.write(' '.join(seq[1:]).replace('[PAD]', '').strip() + '\n')




