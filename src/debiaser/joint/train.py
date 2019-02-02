"""
finetune both models jointly

python joint_train.py --train ../../data/v5/final/bias --test ../../data/v5/final/bias --working_dir TEST/ --train_batch_size 32 --test_batch_size 16 
"""

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


import sys; sys.path.append(".")
from shared.data import get_dataloader
from shared.args import ARGS

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import model as joint_modeling



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


TRAIN_BATCH_SIZE = ARGS.train_batch_size
TEST_BATCH_SIZE = ARGS.test_batch_size


EPOCHS = ARGS.epochs

MAX_SEQ_LEN = ARGS.max_seq_len

CUDA = (torch.cuda.device_count() > 0)
                                                                


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


if ARGS.pretrain_data: 
    pretrain_dataloader, num_pretrain_examples = get_dataloader(
        ARGS.pretrain_data, 
        tok2id, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/pretrain_data.pkl',
        noise=True)

train_dataloader, num_train_examples = get_dataloader(
    ARGS.train,
    tok2id, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/train_data.pkl',
    add_del_tok=ARGS.add_del_tok, 
    tok_dist_path=ARGS.tok_dist_train_path)
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, TEST_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok, 
    tok_dist_path=ARGS.tok_dist_test_path)



# # # # # # # # ## # # # ## # # MODELS # # # # # # # # ## # # # ## # #
if ARGS.no_tok_enrich:
    debias_model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
else:
    debias_model = seq2seq_model.Seq2SeqEnrich(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)


if ARGS.extra_features_top:
    tagging_model= tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    tagging_model= tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    tagging_model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache')
        
if ARGS.tagger_checkpoint is not None and os.path.exists(ARGS.tagger_checkpoint):
    print('LOADING FROM ' + ARGS.tagger_checkpoint)
    tagging_model.load_state_dict(torch.load(ARGS.tagger_checkpoint))
    print('...DONE')



joint_model = joint_modeling.JointModel(
    debias_model=debias_model, tagging_model=tagging_model)


if CUDA:
    joint_model = joint_model.cuda()

model_parameters = filter(lambda p: p.requires_grad, joint_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('NUM PARAMS: ', params)





# # # # # # # # ## # # # ## # # OPTIMIZER, LOSS # # # # # # # # ## # # # ## # #
tagging_loss_fn = tagging_utils.build_loss_fn()
debias_loss_fn, cross_entropy_loss = seq2seq_utils.build_loss_fn(vocab_size=len(tok2id))
optimizer = optim.Adam(
    debias_model.parameters() if ARGS.freeze_tagger else joint_model.parameters(), 
    lr=ARGS.learning_rate)



# # # # # # # # # # # PRETRAINING (optional) # # # # # # # # # # # # # # # #
if ARGS.pretrain_data:
    pretrain_optim = optim.Adam(debias_model.parameters(), lr=ARGS.learning_rate)

    print('PRETRAINING...')
    for epoch in range(ARGS.pretrain_epochs):
        joint_model.train()
        for step, batch in enumerate(tqdm(pretrain_dataloader)):
            if ARGS.debug_skip and step > 2:
                continue

            if CUDA: 
                batch = tuple(x.cuda() for x in batch)
            (
                pre_id, pre_mask, pre_len, 
                post_in_id, post_out_id, 
                pre_tok_label_id, post_tok_label_id, tok_dist,
                replace_id, rel_ids, pos_ids, type_ids, categories
            ) = batch      
            # dont pass tagger args = don't use TAGGER
            post_logits, post_probs, tok_probs, tok_logits = joint_model(
                pre_id, post_in_id, pre_mask, pre_len, tok_dist, type_ids)
            loss = cross_entropy_loss(post_logits, post_out_id, post_tok_label_id)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(joint_model.parameters(), 3.0)
            pretrain_optim.step()
            joint_model.zero_grad()
            



# # # # # # # # # # # # TRAINING # # # # # # # # # # # # # #
writer = SummaryWriter(ARGS.working_dir)

print('INITIAL EVAL...')
joint_model.eval()
hits, preds, golds, srcs = joint_modeling.run_eval(
    joint_model, eval_dataloader, tok2id, WORKING_DIR + '/results_initial.txt',
    MAX_SEQ_LEN, ARGS.beam_width)
writer.add_scalar('eval/bleu', seq2seq_utils.get_bleu(preds, golds), 0)
writer.add_scalar('eval/true_hits', np.mean(hits), 0)

for epoch in range(EPOCHS):
    print('EPOCH ', epoch)
    print('TRAIN...')
    joint_model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA: 
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, post_tok_label_id, tok_dist,
            replace_id, rel_ids, pos_ids, type_ids, categories
        ) = batch      
        post_logits, post_probs, tok_probs, tok_logits = joint_model(
            pre_id, post_in_id, pre_mask, pre_len, tok_dist, type_ids,
            rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)
        loss = debias_loss_fn(post_logits, post_out_id, post_tok_label_id)
        tok_loss = tagging_loss_fn(tok_logits, pre_tok_label_id, apply_mask=pre_tok_label_id)
        loss = loss + (ARGS.tag_loss_mixing_prob * tok_loss)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(joint_model.parameters(), 3.0)
        optimizer.step()
        joint_model.zero_grad()
        
    
    print('SAVING...')
    # joint_model.save(WORKING_DIR + '/model_%d.ckpt' % (epoch+1))

    print('EVAL...')
    joint_model.eval()
    hits, preds, golds, srcs = joint_modeling.run_eval(
        joint_model, eval_dataloader, tok2id, WORKING_DIR + '/results_%d.txt' % epoch,
        MAX_SEQ_LEN, ARGS.beam_width)
    writer.add_scalar('eval/bleu', seq2seq_utils.get_bleu(preds, golds), epoch+1)
    writer.add_scalar('eval/true_hits', np.mean(hits), epoch+1)




