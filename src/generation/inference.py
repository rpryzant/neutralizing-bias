import sys

import json
import data
import models
import numpy as np
import logging
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import evaluation

from cuda import CUDA

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument(
    "--model",
    help="skip training and evaluate",
    type=str
)
args = parser.parse_args()
config = json.load(open(args.config, 'r'))
config['cuda'] = (torch.cuda.device_count() > 0)
working_dir = config['data']['working_dir']

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

with open(os.path.join(working_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/inference_log' % working_dir,
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


print 'Reading data ...'
src_test, tgt_test = data.read_nmt_data(
    src=config['data']['src_test'],
    config=config,
    tgt=config['data']['tgt_test']
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_len']
src_vocab_size = len(src_test['tok2id'])
tgt_vocab_size = len(tgt_test['tok2id'])
softmax_temp = config['model']['self_attn_temp']

weight_mask = torch.ones(tgt_vocab_size)
weight_mask[tgt_test['tok2id']['<pad>']] = 0
if CUDA:
    weight_mask = weight_mask.cuda()
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
if CUDA:
    loss_criterion = loss_criterion.cuda()

torch.manual_seed(config['training']['random_seed'])

model = models.Seq2SeqAttention(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    pad_id_src=src_test['tok2id']['<pad>'],
    pad_id_tgt=tgt_test['tok2id']['<pad>'],
    config=config
)

if ',' in args.model:
    ckpts = args.model.split(',')
elif '-' in args.model:
    parts = args.model.split('-')
    ckpts = [
        working_dir + '/model.%s.ckpt' % i \
        for i in range(int(parts[0]), int(parts[1]))
    ]
else:
    ckpts = [args.model]

for model_ckpt in ckpts:
    model, start_epoch = models.attempt_load_model(
        model=model, 
        checkpoint_path=model_ckpt)
    if CUDA:
        model = model.cuda()
    model.eval()

    logging.info('STARTING INFERENCE: ' + model_ckpt)
    start = time.time()
    bleu, preds, golds = evaluation.evaluate_bleu(
        model, src_test, tgt_test, config, softmax_temp)

    model_id = int(model_ckpt.split('.')[1])

    with open(working_dir + '/preds.%s' % model_id, 'w') as f:
        f.write('\n'.join(preds) + '\n') 
    with open(working_dir + '/golds.%s' % model_id, 'w') as f:
        f.write('\n'.join(golds) + '\n') 
    logging.info('BLEU: %s TIME: %.2fs' % (bleu, (time.time() - start)))


