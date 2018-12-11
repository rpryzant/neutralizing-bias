# -*- coding: utf-8 -*-
"""
train bert 

python multi_bert_trainer.py data/balanced_token/ multi_targeted TEST

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

import sklearn.metrics as metrics

import modeling


data_prefix = sys.argv[1]
mode = sys.argv[2]
working_dir = sys.argv[3]

assert mode in ['multi_always', 'multi_targeted', 'classification', 'tagging_always', 'tagging_targeted']


TRAIN_TEXT = data_prefix + '/text.train'
TRAIN_TOK_LABELS = data_prefix + '/tok_labels.train'
TRAIN_BIAS_LABELS = data_prefix + '/bias_labels.train'

TEST_TEXT = data_prefix + '/text.test'
TEST_TOK_LABELS = data_prefix + '/tok_labels.test'
TEST_BIAS_LABELS = data_prefix + '/bias_labels.test'

WORKING_DIR = working_dir

NUM_BIAS_LABELS = 2
NUM_TOK_LABELS = 3

BERT_MODEL = "bert-base-uncased"

TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2

EPOCHS = 15

LEARNING_RATE = 5e-5

MAX_SEQ_LEN = 100

CUDA = (torch.cuda.device_count() > 0)

if CUDA:
    print('USING CUDA')


def get_examples(text_path, tok_labels_path, bias_labels_path, tokenizer, possible_labels, max_seq_len):
    label2id = {label: i for i, label in enumerate(possible_labels)}
    label2id['mask'] = len(label2id)

    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    skipped = 0 

    out = {'tokens': [], 'input_ids': [], 'segment_ids': [], 'input_masks': [], 'tok_label_ids': [], 'bias_label_ids': []}

    for i, (line, tok_labels, bias_label) in enumerate(tqdm(zip(open(text_path), open(tok_labels_path), open(bias_labels_path)))):
        # ignore the unbiased sentences with tagging -- TODO -- toggle this?    
        tokens = line.strip().split() # Pre-tokenized
        tok_labels = tok_labels.strip().split()

        assert len(tokens) == len(tok_labels)

        # account for [CLS] and [SEP]
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]
            tok_labels = tok_labels[:max_seq_len - 2]

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        try:
            input_ids = pad(tokenizer.convert_tokens_to_ids(tokens), 0)
        except KeyError:
            # TODO FUCK THIS ENCODING BUG!!!
            skipped += 1
            continue


        segment_ids = pad([0 for _ in range(len(tokens))], 0)
        input_mask = pad([1] * len(tokens), 0)
        # Mask out [CLS] token in front too
        tok_label_ids = pad([label2id['mask']] + [label2id[l] for l in tok_labels], label2id['mask'])
        bias_label_id = label2id[bias_label.strip()]

        assert len(input_ids) == len(segment_ids) == len(input_mask) == len(tok_label_ids) == max_seq_len

        out['tokens'].append(tokens)
        out['input_ids'].append(input_ids)
        out['segment_ids'].append(segment_ids)
        out['input_masks'].append(input_mask)
        out['tok_label_ids'].append(tok_label_ids)
        out['bias_label_ids'].append(bias_label_id)

    print('SKIPPED ', skipped)
    return out


def get_dataloader(data_path, tok_labels_path, bias_labels_path, tokenizer, batch_size, pickle_path=None):
    if pickle_path is not None and os.path.exists(pickle_path):
        train_examples = pickle.load(open(pickle_path, 'rb'))
    else:
        train_examples = get_examples(
            text_path=data_path, 
            tok_labels_path=tok_labels_path,
            bias_labels_path=bias_labels_path,
            tokenizer=tokenizer,
            possible_labels=["0", "1"],
            max_seq_len=MAX_SEQ_LEN)
        pickle.dump(train_examples, open(pickle_path, 'wb'))

    train_data = TensorDataset(
        torch.tensor(train_examples['input_ids'], dtype=torch.long),
        torch.tensor(train_examples['input_masks'], dtype=torch.long),
        torch.tensor(train_examples['segment_ids'], dtype=torch.long),
        torch.tensor(train_examples['bias_label_ids'], dtype=torch.long),
        torch.tensor(train_examples['tok_label_ids'], dtype=torch.long))

    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=batch_size)

    return train_dataloader, len(train_examples['input_ids'])

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

        'bias_loss': [],
        'bias_logits': [],
        'bias_labels': [],
        
        'tok_loss': [],
        'tok_logits': [],
        'tok_labels': [],

        'classification_hits': [],
        'labeling_hits': []
    }

    for step, batch in enumerate(tqdm(eval_dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        input_ids, input_mask, segment_ids, bias_label_ids, tok_label_ids = batch

        with torch.no_grad():
            bias_logits, tok_logits = model(input_ids, segment_ids, input_mask)
            bias_loss = cls_criterion(bias_logits.view(-1, NUM_BIAS_LABELS), bias_label_ids.view(-1))
            tok_loss = tok_criterion(tok_logits.view(-1, NUM_TOK_LABELS), tok_label_ids.view(-1))
            
        out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in input_ids.cpu().numpy()]
        
        out['bias_loss'].append(float(bias_loss.cpu().numpy()))
        out['bias_logits'] += bias_logits.detach().cpu().numpy().tolist()
        out['bias_labels'] += bias_label_ids.cpu().numpy().tolist()


        out['tok_loss'].append(float(tok_loss.cpu().numpy()))
        out['tok_logits'] += tok_logits.detach().cpu().numpy().tolist()
        out['tok_labels'] += tok_label_ids.cpu().numpy().tolist()

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

def tag_accuracy(logits, bias_labels, tok_labels, top=1):
    probs = softmax(np.array(logits)[:, :, : NUM_TOK_LABELS - 1], axis=2)

    hits = [
        is_ranking_hit(prob_dist, tok_label, top=top) 
        for prob_dist, tok_label, bias_label in zip(probs, tok_labels, bias_labels)
        if bias_label == 1
    ]
    return sum(hits) * 1.0 / len(hits)


print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
train_dataloader, num_train_examples = get_dataloader(
    TRAIN_TEXT, TRAIN_TOK_LABELS, TRAIN_BIAS_LABELS, 
    tokenizer, TRAIN_BATCH_SIZE, WORKING_DIR + '/train_data.pkl')
eval_dataloader, num_eval_examples = get_dataloader(
    TEST_TEXT, TEST_TOK_LABELS, TEST_BIAS_LABELS,
    tokenizer, TEST_BATCH_SIZE, WORKING_DIR + '/test_data.pkl')

print('BUILDING MODEL...')
model = modeling.BertForMultitask.from_pretrained(
    BERT_MODEL,
    cls_num_labels=NUM_BIAS_LABELS,
    tok_num_labels=NUM_TOK_LABELS,
    cache_dir=WORKING_DIR + '/cache')
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
bias_acc = classification_accuracy(results['bias_logits'], results['bias_labels'])
tok_acc = tag_accuracy(results['tok_logits'], results['bias_labels'], results['tok_labels'], top=1)
writer.add_scalar('eval/bias_loss', np.mean(results['bias_loss']), 0)
writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
writer.add_scalar('eval/bias_acc', bias_acc, 0)
writer.add_scalar('eval/tok_loss', tok_acc, 0)

print('TRAINING...')
model.train()
train_step = 0
for epoch in range(EPOCHS):
    print('STARTING EPOCH ', epoch)
    for step, batch in enumerate(tqdm(train_dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        input_ids, input_mask, segment_ids, bias_label_ids, tok_label_ids = batch
        bias_logits, tok_logits = model(input_ids, segment_ids, input_mask)
        bias_loss = cls_criterion(bias_logits.view(-1, NUM_BIAS_LABELS), bias_label_ids.view(-1))

        # only backprop on biased examples if so
        tok_losses = []
        for tok_log, tok_lab, bias_lab in zip(tok_logits, tok_label_ids, bias_label_ids):
            if mode in ['multi_targeted', 'tagging_targeted'] and int(bias_lab.detach().cpu().numpy()) == 0:
                continue
            tok_losses.append(tok_criterion(tok_log.view(-1, NUM_TOK_LABELS), tok_lab.view(-1)))
        if not tok_losses:
            tok_loss = torch.tensor(0.0)
        else:
            tok_loss = sum(tok_losses) / len(tok_losses)
    
        if mode in ['multi_always', 'multi_targeted']:
            loss = bias_loss + tok_loss
        elif mode == 'classification':
            loss = bias_loss
        elif mode in ['tagging_always', 'tagging_targeted']:
            loss = tok_loss

        loss.backward()
        optimizer.step()
        model.zero_grad()

        writer.add_scalar('train/bias_loss', bias_loss.data[0], train_step)
        writer.add_scalar('train/tok_loss', tok_loss.data[0], train_step)
        writer.add_scalar('train/loss', loss.data[0], train_step)
        train_step += 1

    # eval
    print('EVAL...')
    model.eval()
    results = run_inference(model, eval_dataloader, cls_criterion, tok_criterion)
    bias_acc = classification_accuracy(results['bias_logits'], results['bias_labels'])
    tok_acc = tag_accuracy(results['tok_logits'], results['bias_labels'], results['tok_labels'], top=1)
    writer.add_scalar('eval/bias_loss', np.mean(results['bias_loss']), 0)
    writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
    writer.add_scalar('eval/bias_acc', bias_acc, 0)
    writer.add_scalar('eval/tok_loss', tok_acc, 0)

    model.train()



