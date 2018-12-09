"""
train bert 
"""
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
import torch
import pickle
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter

import sklearn.metrics as metrics

TRAIN_DATA = "/home/rpryzant/persuasion/src/discrimination/data/data/text.train"
TRAIN_LABELS = "/home/rpryzant/persuasion/src/discrimination/data/data/labels.train"

TEST_DATA = "/home/rpryzant/persuasion/src/discrimination/data/data/text.test"
TEST_LABELS = "/home/rpryzant/persuasion/src/discrimination/data/data/labels.test"

NUM_LABELS=2

BERT_MODEL = "bert-base-uncased"

WORKING_DIR = "test"

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 16
EPOCHS = 15

LEARNING_RATE = 5e-5

MAX_SEQ_LEN = 60

CUDA = (torch.cuda.device_count() > 0)

if CUDA:
    print('USING CUDA')


def get_examples(text_path, labels_path, tokenizer, possible_labels, max_seq_len):
    label2id = {label: i for i, label in enumerate(possible_labels)}

    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    out = {'tokens': [], 'input_ids': [], 'segment_ids': [], 'input_masks': [], 'label_ids': []}

    for i, (line, label) in enumerate(tqdm(zip(open(text_path), open(labels_path)))):
        tokens = tokenizer.tokenize(line.strip())
        # account for [CLS] and [SEP]
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = pad(tokenizer.convert_tokens_to_ids(tokens), 0)
        segment_ids = pad([0 for _ in range(len(tokens))], 0)
        input_mask = pad([1] * len(tokens), 0)
        label_id = label2id[label.strip()]

        assert len(input_ids) == len(segment_ids) == len(input_mask) == max_seq_len

        out['tokens'].append(tokens)
        out['input_ids'].append(input_ids)
        out['segment_ids'].append(segment_ids)
        out['input_masks'].append(input_mask)
        out['label_ids'].append(label_id)

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

    eval_probs = softmax(np.array(eval_logits), axis=1)
    eval_preds = np.argmax(eval_probs, axis=1)
    eval_labels = np.array(eval_label_ids)

    return eval_probs, eval_preds, eval_labels, eval_loss

print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
train_dataloader, num_train_examples = get_dataloader(
    DATA_PREFIX + '.train', LABELS_PREFIX + '.train', tokenizer, TRAIN_BATCH_SIZE, WORKING_DIR + '/train_data.pkl')
eval_dataloader, num_eval_examples = get_dataloader(
    DATA_PREFIX + '.test', LABELS_PREFIX + '.test', tokenizer, EVAL_BATCH_SIZE, WORKING_DIR + '/test_data.pkl')

print('BUILDING MODEL...')
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL, 
    cache_dir=WORKING_DIR + '/cache')
if CUDA:
    model = model.cuda()

print('PREPPING RUN...')
optimizer = make_optimizer(model, int((num_train_examples * EPOCHS) / TRAIN_BATCH_SIZE))
criterion = CrossEntropyLoss()
writer = SummaryWriter(WORKING_DIR)


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
    eval_probs, eval_preds, eval_labels, eval_loss = run_inference(model, eval_dataloader)
    model.train()
    
    writer.add_scalar('eval/loss', np.mean(eval_loss), epoch)
    writer.add_scalar('eval/acc', metrics.accuracy_score(eval_labels, eval_preds), epoch)
    writer.add_scalar('eval/precision', metrics.precision_score(eval_labels, eval_preds), epoch)
    writer.add_scalar('eval/recall', metrics.recall_score(eval_labels, eval_preds), epoch)
    writer.add_scalar('eval/f1', metrics.f1_score(eval_labels, eval_preds), epoch)
    writer.add_scalar('eval/auc', metrics.roc_auc_score(eval_labels, eval_probs[:, 1]), epoch)





