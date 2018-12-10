# -*- coding: utf-8 -*-
"""
train bert 

python train_bert_tagger.py data_all/ test
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

data_prefix = sys.argv[1]
working_dir = sys.argv[2]

TRAIN_DATA = data_prefix + "/text.train"
TRAIN_LABELS = data_prefix + "/labels.train"

TEST_DATA = data_prefix + "/text.test"
TEST_LABELS = data_prefix + "/labels.test"

WORKING_DIR = working_dir

NUM_LABELS = 3

BERT_MODEL = "bert-base-uncased"

TRAIN_BATCH_SIZE = 36
EVAL_BATCH_SIZE = 16
EPOCHS = 15

LEARNING_RATE = 5e-5

MAX_SEQ_LEN = 100

CUDA = (torch.cuda.device_count() > 0)

if CUDA:
    print('USING CUDA')


def get_examples(text_path, labels_path, tokenizer, possible_labels, max_seq_len):
    label2id = {label: i for i, label in enumerate(possible_labels)}
    label2id['mask'] = len(label2id)

    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    skipped = 0 

    out = {'tokens': [], 'input_ids': [], 'segment_ids': [], 'input_masks': [], 'label_ids': []}

    for i, (line, labels) in enumerate(tqdm(zip(open(text_path), open(labels_path)))):
        tokens = line.strip().split() # Pre-tokenized
        labels = labels.strip().split()

        assert len(tokens) == len(labels)

        # account for [CLS] and [SEP]
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]
            labels = labels[:max_seq_len - 2]

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
        label_ids = pad([label2id['mask']] + [label2id[l] for l in labels], label2id['mask'])

        assert len(input_ids) == len(segment_ids) == len(input_mask) == max_seq_len

        out['tokens'].append(tokens)
        out['input_ids'].append(input_ids)
        out['segment_ids'].append(segment_ids)
        out['input_masks'].append(input_mask)
        out['label_ids'].append(label_ids)

    print('SKIPPED ', skipped)
    return out


def get_dataloader(data_path, labels_path, tokenizer, batch_size, pickle_path=None):
    if pickle_path is not None and os.path.exists(pickle_path):
        train_examples = pickle.load(open(pickle_path, 'rb'))
    else:
        train_examples = get_examples(
            text_path=data_path, 
            labels_path=labels_path,
            tokenizer=tokenizer,
            possible_labels=["0", "1"],
            max_seq_len=MAX_SEQ_LEN)
        pickle.dump(train_examples, open(pickle_path, 'wb'))

    train_data = TensorDataset(
        torch.tensor(train_examples['input_ids'], dtype=torch.long),
        torch.tensor(train_examples['input_masks'], dtype=torch.long),
        torch.tensor(train_examples['segment_ids'], dtype=torch.long),
        torch.tensor(train_examples['label_ids'], dtype=torch.long))

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

def is_ranking_hit(probs, labels, top=1):
    probs_indices = list(zip(probs[:, 1], range(len(labels))))
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))

    if sum([labels[i] for i in top_indices]) > 0:
        return 1
    else:
        return 0

def run_inference(model, eval_dataloader):
    eval_logits = []
    eval_loss = []
    eval_label_ids = []
    eval_input_toks = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits.view(-1, NUM_LABELS), label_ids.view(-1))

        eval_logits += logits.detach().cpu().numpy().tolist()
        eval_label_ids += label_ids.cpu().numpy().tolist() 
        eval_input_toks += [tokenizer.convert_ids_to_tokens(seq) for seq in input_ids.cpu().numpy()]
        eval_loss.append(float(loss.cpu().numpy()))

    # only take stuff on 1's
    eval_probs, eval_preds, eval_labels, ranking_hits = [], [], [], []
    for logits, labels in zip(eval_logits, eval_label_ids):
        # get rid of padding stuff
        seq_len = list(labels[1:]).index(NUM_LABELS - 1)
        labels = labels[1:seq_len + 1]
        logits = np.array(logits[1:seq_len + 1])[:, :2]
        probs = softmax(logits, axis=1)
        preds = np.argmax(probs, axis=1)

        ranking_hit = is_ranking_hit(probs, labels, top=1)

        eval_probs += list(probs)
        eval_preds += list(preds)
        eval_labels += list(labels)
        ranking_hits += list([ranking_hit])

    eval_probs = np.array(eval_probs)
    eval_preds = np.array(eval_preds)
    eval_labels = np.array(eval_labels)

    return eval_probs, eval_preds, eval_labels, eval_loss, ranking_hits

print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
train_dataloader, num_train_examples = get_dataloader(
    TRAIN_DATA, TRAIN_LABELS, tokenizer, TRAIN_BATCH_SIZE, WORKING_DIR + '/train_data.pkl')
eval_dataloader, num_eval_examples = get_dataloader(
    TEST_DATA, TEST_LABELS, tokenizer, EVAL_BATCH_SIZE, WORKING_DIR + '/test_data.pkl')

print('BUILDING MODEL...')
model = BertForTokenClassification.from_pretrained(
    BERT_MODEL,
    num_labels=NUM_LABELS, # +1 for pad?
    cache_dir=WORKING_DIR + '/cache')
if CUDA:
    model = model.cuda()

print('PREPPING RUN...')
optimizer = make_optimizer(model, int((num_train_examples * EPOCHS) / TRAIN_BATCH_SIZE))
weight_mask = torch.ones(NUM_LABELS)
weight_mask[-1] = 0
if CUDA:
    weight_mask = weight_mask.cuda()
    criterion = CrossEntropyLoss(weight=weight_mask).cuda()
else:
    criterion = CrossEntropyLoss(weight=weight_mask)

writer = SummaryWriter(WORKING_DIR)


print('INITIAL EVAL...')
model.eval()
eval_probs, eval_preds, eval_labels, eval_loss, ranking_hits = run_inference(model, eval_dataloader)
writer.add_scalar('eval/loss', np.mean(eval_loss), 0)
writer.add_scalar('eval/acc', metrics.accuracy_score(eval_labels, eval_preds), 0)
writer.add_scalar('eval/precision', metrics.precision_score(eval_labels, eval_preds), 0)
writer.add_scalar('eval/recall', metrics.recall_score(eval_labels, eval_preds), 0)
writer.add_scalar('eval/f1', metrics.f1_score(eval_labels, eval_preds), 0)
writer.add_scalar('eval/ranking_acc', sum(ranking_hits) * 1.0 / len(ranking_hits), 0)

print('TRAINING...')
model.train()
train_step = 0
for epoch in range(EPOCHS):
    print('STARTING EPOCH ', epoch)
    for step, batch in enumerate(tqdm(train_dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits = model(input_ids, segment_ids, input_mask)
        loss = criterion(logits.view(-1, NUM_LABELS), label_ids.view(-1))
        loss.backward()
        optimizer.step()
        model.zero_grad()

        writer.add_scalar('train/loss', loss.data[0], train_step)
        train_step += 1

    # eval
    print('EVAL...')
    model.eval()
    eval_probs, eval_preds, eval_labels, eval_loss, ranking_hits = run_inference(model, eval_dataloader)
    writer.add_scalar('eval/loss', np.mean(eval_loss), epoch + 1)
    writer.add_scalar('eval/acc', metrics.accuracy_score(eval_labels, eval_preds), epoch + 1)
    writer.add_scalar('eval/precision', metrics.precision_score(eval_labels, eval_preds), epoch + 1)
    writer.add_scalar('eval/recall', metrics.recall_score(eval_labels, eval_preds), epoch + 1)
    writer.add_scalar('eval/f1', metrics.f1_score(eval_labels, eval_preds), epoch + 1)
    writer.add_scalar('eval/ranking_acc', sum(ranking_hits) * 1.0 / len(ranking_hits), epoch + 1)
    # doesn't make sense cause we only have positive labels
    # writer.add_scalar('eval/auc', metrics.roc_auc_score(eval_labels, eval_probs[:, 1]), epoch + 1)


    model.train()



