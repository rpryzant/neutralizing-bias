# -*- coding: utf-8 -*-
"""
train bert

python tagging/train.py --train ../../data/v6/corpus.wordbiased.tag.train --test ../../data/v6/corpus.wordbiased.tag.test --working_dir TEST --train_batch_size 3 --test_batch_size 10  --hidden_size 32 --debug_skip

for training on the smaller 100 test dataset
python tagging/train.py --train ../../data/v6/corpus.wordbiased.tag.train --test ../../data/100sample.tsv --working_dir ATTENTION --train_batch_size 3 --test_batch_size 10  --hidden_size 32 #--debug_skip

For actual run
python tagging/train.py --train ../../data/corpus.wordbiased.tag.train --test ../../data/100sample.tsv --working_dir ATTENTION --train_batch_size 32 --test_batch_size 10

# For training on trash
python tagging/train.py --train ../../data/100sample.tsv --test ../../data/100sample.tsv --working_dir attention_overfit_noattention --train_batch_size 32 --test_batch_size 10


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

import model as tagging_model
import utils as tagging_utils

import sys; sys.path.append('.')
from shared.data import get_dataloader
from shared.args import ARGS
from shared.constants import CUDA


if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)

with open(ARGS.working_dir + '/command.sh', 'w') as f:
    f.write('python' + ' '.join(sys.argv) + '\n')


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #

print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

print(ARGS)
train_dataloader, num_train_examples = get_dataloader(
    ARGS.train,
    tok2id, ARGS.train_batch_size,
    ARGS.working_dir + '/train_data.pkl',
    categories_path=ARGS.categories_file)
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, ARGS.test_batch_size, ARGS.working_dir + '/test_data.pkl',
    test=True, categories_path=ARGS.categories_file)

# # # # # # # # ## # # # ## # # MODEL # # # # # # # # ## # # # ## # #


print('BUILDING MODEL...')
if ARGS.extra_features_top:
    model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    model = tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache',
        tok2id=tok2id)
if CUDA:
    model = model.cuda()
    print("cuda available")

print('PREPPING RUN...')

# # # # # # # # ## # # # ## # # OPTIMIZER, LOSS # # # # # # # # ## # # # ## # #


optimizer = tagging_utils.build_optimizer(
    model, int((num_train_examples * ARGS.epochs) / ARGS.train_batch_size),
    ARGS.learning_rate)

loss_fn = tagging_utils.build_loss_fn()

# # # # # # # # ## # # # ## # # TRAIN # # # # # # # # ## # # # ## # #

writer = SummaryWriter(ARGS.working_dir)


print('INITIAL EVAL...')
model.eval()
results = tagging_utils.run_inference(model, eval_dataloader, loss_fn, tokenizer)
writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), 0)

attention_results = []

def output_attention_results(results, epoch):
    global attention_results
    '''
    Write out the results of the attention layer for interpretable AI.

    Results is a dictionary that contains the following important entries:
    - results['input_toks']
    - results['post_toks']
    - results['tok_labels']
    - results['tok_probs']
    - results['attention_dist']
    '''

    batch_labels = results['tok_labels']
    batch_logits = results['tok_logits']
    for i, labels in enumerate(batch_labels):
        try:
            idx_biased = labels.index(1)
        except:
            continue

        # checking if correct -- fuck it just printing out everyone and appending
        #if np.argmax(batch_logits[i][idx_biased]) == 1:
        #    print("in here")

        attention_result = {}
        attention_result['labels'] = labels
        attention_result['input_toks'] = results['input_toks'][i]
        attention_result['bias_word'] = results['input_toks'][i][idx_biased]
        attention_result['attention_dist'] = results['attention_dist'][i][idx_biased]
        attention_result['tok_logits'] = batch_logits[i]
        attention_result['epoch_num'] = epoch
        attention_results.append(attention_result)

print('TRAINING...')
model.train()
for epoch in range(ARGS.epochs):
    print('STARTING EPOCH ', epoch)
    losses = tagging_utils.train_for_epoch(model, train_dataloader, loss_fn, optimizer)
    writer.add_scalar('train/loss', np.mean(losses), epoch + 1)

    # eval
    print('EVAL...')
    model.eval()
    results = tagging_utils.run_inference(model, eval_dataloader, loss_fn, tokenizer)
    print(np.mean(results['labeling_hits']))
    print(np.mean(results['tok_loss']))
    writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), epoch + 1)
    writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), epoch + 1)

    model.train()

    output_attention_results(results, epoch)

    print('SAVING...')
    torch.save(model.state_dict(), ARGS.working_dir + '/model_%d.ckpt' % epoch)
    pickle.dump(attention_results, open(ARGS.working_dir + "/attention_results_epoch_{}.pkl".format(epoch), 'wb+'))

    #srun --partition=jag-lo --gres=gpu:1 --mem=16G --cpus-per-task=4 --pty bash

print("saving out attention results")
