# -*- coding: utf-8 -*-
"""
train bert 

python multi_bert_trainer.py data/balanced_token/ multi_targeted TEST
python word_replacer.py /home/rpryzant/persuasion/data/v4/tok/biased /home/rpryzant/persuasion/data/v4/tok/biased multi_cls TEST
"""
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import pickle
import sys
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from simplediff import diff
import sklearn.metrics as metrics

import modeling


train_data_prefix = sys.argv[1]
test_data_prefix = sys.argv[2]
mode = sys.argv[3]
working_dir = sys.argv[4]
if not os.path.exists(working_dir):
    os.makedirs(working_dir)


assert mode in [
    'seperate_cls', 'seperate_tok', 'multi_from_cls', 'multi_from_tok',
    'multi_tok_attn'
]


TRAIN_TEXT = train_data_prefix + '.train.pre'
TRAIN_TEXT_POST = train_data_prefix + '.train.post'
TRAIN_TOK_LABELS = train_data_prefix + '.train.tok_labels'
TRAIN_BIAS_LABELS = train_data_prefix + '.train.seq_labels'

TEST_TEXT = test_data_prefix + '.test.pre'
TEST_TEXT_POST = test_data_prefix + '.test.post'
TEST_TOK_LABELS = test_data_prefix + '.test.tok_labels'
TEST_BIAS_LABELS = test_data_prefix + '.test.seq_labels'

WORKING_DIR = working_dir

NUM_BIAS_LABELS = 2
NUM_TOK_LABELS = 3

BERT_MODEL = "bert-base-uncased"

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16

EPOCHS = 2

LEARNING_RATE = 5e-5

MAX_SEQ_LEN = 5#100

CUDA = (torch.cuda.device_count() > 0)

if CUDA:
    print('USING CUDA')


def get_tok_labels(s_diff):
    tok_labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            tok_labels += ['0'] * len(chunk)
        elif tag == '-':
            tok_labels += ['1'] * len(chunk)
        else:
            pass

    return tok_labels


def get_examples(text_path, text_post_path, tok_labels_path, bias_labels_path, tokenizer, possible_labels, max_seq_len):
    label2id = {label: i for i, label in enumerate(possible_labels)}
    label2id['mask'] = len(label2id)

    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    skipped = 0 

    out = {
        'tokens': [], 'input_ids': [], 'segment_ids': [], 
        'input_masks': [], 'tok_label_ids': [], 'bias_label_ids': [],
        'replacement_word_labels': []
    }

    for i, (line, post_line, tok_labels, bias_label) in enumerate(tqdm(zip(open(text_path), open(text_post_path), open(tok_labels_path), open(bias_labels_path)))):
        # ignore the unbiased sentences with tagging -- TODO -- toggle this?    
        tokens = line.strip().split() # Pre-tokenized
        post_tokens = post_line.strip().split()
        tok_diff = diff(tokens, post_tokens)
        tok_labels = get_tok_labels(tok_diff)
        assert len(tokens) == len(tok_labels)

        try:
            replace_token = next( (chunk for tag, chunk in tok_diff if tag == '+') )
            replace_id = tokenizer.convert_tokens_to_ids(replace_token)[0]
        except StopIteration:
            replace_token = None
            replace_id = len(tokenizer.vocab)
        except KeyError:
            skipped += 1
            continue

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

        if 1 not in tok_label_ids: # make sure its a real edit
            skipped += 1
            continue

        assert len(input_ids) == len(segment_ids) == len(input_mask) == len(tok_label_ids) == max_seq_len

        out['tokens'].append(tokens)
        out['input_ids'].append(input_ids)
        out['segment_ids'].append(segment_ids)
        out['input_masks'].append(input_mask)
        out['tok_label_ids'].append(tok_label_ids)
        out['bias_label_ids'].append(bias_label_id)
        out['replacement_word_labels'].append(replace_id)

    print('SKIPPED ', skipped)
    return out


def get_dataloader(data_path, post_data_path, tok_labels_path, bias_labels_path, tokenizer, batch_size, pickle_path=None, test=False):
    if pickle_path is not None and os.path.exists(pickle_path):
        train_examples = pickle.load(open(pickle_path, 'rb'))
    else:
        train_examples = get_examples(
            text_path=data_path, 
            text_post_path=post_data_path,
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
        torch.tensor(train_examples['tok_label_ids'], dtype=torch.long),
        torch.tensor(train_examples['replacement_word_labels'], dtype=torch.long))

    train_dataloader = DataLoader(
        train_data,
        sampler=(SequentialSampler(train_data) if test else RandomSampler(train_data)),
        batch_size=batch_size)

    return train_dataloader, len(train_examples['input_ids'])

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def run_inference(model, eval_dataloader, cls_criterion, tok_criterion, tokenizer=None, tok_preds=None):
    out = {
        'input_toks': [],

        'replacement_loss': [],
        'replacement_logits': [],
        'replacement_labels': [],
        
        'tok_loss': [],
        'tok_logits': [],
        'tok_labels': [],

        'bias_labels': []
    }
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        input_ids, input_mask, segment_ids, bias_label_ids, tok_label_ids, replace_ids = batch

        with torch.no_grad():
            if tok_preds is None:
                bias_indices = np.where(tok_label_ids.cpu().numpy() == 1)[1]
            else:
                batch_size = eval_dataloader.batch_size
                bias_indices = tok_preds[batch_size * step : (batch_size * step) + batch_size]

            replacement_logits, tok_logits, attn_probs = model(input_ids, segment_ids, input_mask, tok_indices=bias_indices)
            replacement_loss = cls_criterion(replacement_logits.view(-1, len(tokenizer.vocab)+1), replace_ids.view(-1))
            tok_loss = tok_criterion(tok_logits.view(-1, NUM_TOK_LABELS), tok_label_ids.view(-1))
            
        out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in input_ids.cpu().numpy()]
        
        out['replacement_loss'].append(float(replacement_loss.cpu().numpy()))
        out['replacement_logits'] += replacement_logits.detach().cpu().numpy().tolist()
        out['replacement_labels'] += replace_ids.cpu().numpy().tolist()

        out['bias_labels'] += bias_label_ids.detach().cpu().numpy().tolist()

        out['tok_loss'].append(float(tok_loss.cpu().numpy()))
        # mask logits for metric stuff. also only take the class we're interested in (positive)
        tok_logits = tok_logits.detach().cpu().numpy()[:, :, 1] + (1.0 - input_mask.detach().cpu().numpy()) * -10000.0
        out['tok_logits'] += tok_logits.tolist()
        out['tok_labels'] += tok_label_ids.cpu().numpy().tolist()

    model.train()

    return out

def classification_accuracy(logits, labels, prior=None):
    probs = softmax(np.array(logits), axis=1)
    preds = np.argmax(probs, axis=1)
    acc = sum([
        1 if (pi == li and priori == 1) else 0 
        for (pi, li, priori) in zip(preds, labels, (prior or [1] * len(preds)))
    ]) * 1.0 / len(preds)
    return acc

def is_ranking_hit(probs, labels, top=1):
    # get rid of padding idx
    [probs, labels] = list(zip(*[(p, l)  for p, l in zip(probs, labels) if l != NUM_TOK_LABELS - 1 ]))
    probs_indices = list(zip(np.array(probs), range(len(labels))))
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    if sum([labels[i] for i in top_indices]) > 0:
        return 1
    else:
        return 0

def tag_accuracy(logits, bias_labels, tok_labels, top=1):
    probs = softmax(np.array(logits), axis=1)
    hits = [
        is_ranking_hit(prob_dist, tok_label, top=top) 
        for prob_dist, tok_label, bias_label in zip(probs, tok_labels, bias_labels)
        if bias_label == 1
    ]
    return sum(hits) * 1.0 / len(hits), hits


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


def train(model, mode, train_dataloader, cls_criterion, tok_criterion, writer, epochs, num_train_steps, num_tok_labels, num_replacement_labels, name=''):
    optimizer = make_optimizer(model, num_train_steps)

    model.train()
    train_step = 0
    for epoch in range(epochs):
        print('STARTING EPOCH ', epoch)
        for step, batch in enumerate(train_dataloader):
            if step > 3:
                continue
            if CUDA:
                batch = tuple(x.cuda() for x in batch)
            input_ids, input_mask, segment_ids, bias_label_ids, tok_label_ids, replace_ids = batch

            bias_indices = np.where(tok_label_ids.cpu().numpy() == 1)[1]

            replace_logits, tok_logits, attn_probs = model(input_ids, segment_ids, input_mask, tok_indices=bias_indices)

            replacement_loss = cls_criterion(replace_logits.view(-1, num_replacement_labels), replace_ids.view(-1))
            tok_loss = tok_criterion(tok_logits.view(-1, num_tok_labels), tok_label_ids.view(-1))

            if mode == 'multi':
                loss = replacement_loss + tok_loss
            elif mode == 'replace':
                loss = replacement_loss
            elif mode == 'tok':
                loss = tok_loss
            else:
                raise Exception('unknown train mode')

            loss.backward()
            optimizer.step()
            model.zero_grad()

            writer.add_scalar('train/%s_replacement_loss' % name, replacement_loss.data[0], train_step)
            writer.add_scalar('train/%s_tok_loss' % name, tok_loss if isinstance(tok_loss, int) else tok_loss.data[0], train_step)
            writer.add_scalar('train/%s_loss' % name, loss.data[0], train_step)
            train_step += 1
    

# def write_results()

print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
num_replacement_labels = len(tokenizer.vocab) + 1

train_dataloader, num_train_examples = get_dataloader(
    TRAIN_TEXT, TRAIN_TEXT_POST, TRAIN_TOK_LABELS, TRAIN_BIAS_LABELS, 
    tokenizer, TRAIN_BATCH_SIZE, WORKING_DIR + '/train_data.pkl')
writer = SummaryWriter(WORKING_DIR)
num_train_steps = int((num_train_examples * EPOCHS) / TRAIN_BATCH_SIZE)



print('BUILDING MODEL...')
if mode == 'multi_from_tok':
    tok_model = modeling.BertForReplacementTOK.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        replace_num_labels=num_replacement_labels,
        cache_dir=WORKING_DIR + '/cache')
    replace_model = tok_model
elif mode == 'multi_from_cls':
    tok_model = modeling.BertForReplacementCLS.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        replace_num_labels=num_replacement_labels,
        cache_dir=WORKING_DIR + '/cache')
    replace_model = tok_model
elif mode == 'seperate_tok':
    tok_model = modeling.BertForReplacementTOK.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        replace_num_labels=num_replacement_labels,
        cache_dir=WORKING_DIR + '/cache')
    replace_model = modeling.BertForReplacementTOK.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        replace_num_labels=num_replacement_labels,
        cache_dir=WORKING_DIR + '/cache')
elif mode == 'seperate_cls':
    tok_model = modeling.BertForReplacementCLS.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        replace_num_labels=num_replacement_labels,
        cache_dir=WORKING_DIR + '/cache')
    replace_model = modeling.BertForReplacementCLS.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        replace_num_labels=num_replacement_labels,
        cache_dir=WORKING_DIR + '/cache')
elif mode == 'multi_tok_attn':
    tok_model = modeling.BertForReplacementTOKAttnClassifier.from_pretrained(
        BERT_MODEL,
        cls_num_labels=NUM_BIAS_LABELS,
        tok_num_labels=NUM_TOK_LABELS,
        replace_num_labels=num_replacement_labels,
        cache_dir=WORKING_DIR + '/cache',
        attn_type='bert')
    replace_model = tok_model
else:
    raise Exception("unknown mode type:", mode)

if CUDA:
    tok_model = tok_model.cuda()
    replace_model = replace_model.cuda()

weight_mask = torch.ones(NUM_TOK_LABELS)
weight_mask[-1] = 0
if CUDA:
    weight_mask = weight_mask.cuda()
    tok_criterion = CrossEntropyLoss(weight=weight_mask).cuda()
    cls_criterion = CrossEntropyLoss()
else:
    tok_criterion = CrossEntropyLoss(weight=weight_mask)
    cls_criterion = CrossEntropyLoss()

if 'multi' in mode:
    train(tok_model, 'multi', train_dataloader, cls_criterion, tok_criterion, writer, EPOCHS, num_train_steps,
        num_tok_labels=NUM_TOK_LABELS,
        num_replacement_labels=num_replacement_labels)
else:
    train(tok_model, 'tok', train_dataloader, cls_criterion, tok_criterion, writer, EPOCHS, num_train_steps,
        num_tok_labels=NUM_TOK_LABELS,
        num_replacement_labels=num_replacement_labels, name='tok_model')
    train(replace_model, 'replace', train_dataloader, cls_criterion, tok_criterion, writer, EPOCHS, num_train_steps,
        num_tok_labels=NUM_TOK_LABELS,
        num_replacement_labels=num_replacement_labels, name='replace_model')


eval_dataloader, num_eval_examples = get_dataloader(
    TEST_TEXT, TEST_TEXT_POST, TEST_TOK_LABELS, TEST_BIAS_LABELS,
    tokenizer, TEST_BATCH_SIZE, WORKING_DIR + '/test_data.pkl',
    test=True)

# tok results
tok_results = run_inference(tok_model, eval_dataloader, cls_criterion, tok_criterion, tokenizer)
tok_acc, tok_hits = tag_accuracy(tok_results['tok_logits'], tok_results['bias_labels'], tok_results['tok_labels'], top=1)
writer.add_scalar('eval/tok_acc', tok_acc, 0)
writer.add_scalar('eval/tok_loss', np.mean(tok_results['tok_loss']), 0)

# raw replace results (as if there was no tok prediction)
raw_replacement_results = run_inference(replace_model, eval_dataloader, cls_criterion, tok_criterion, tokenizer)
raw_replacement_acc = classification_accuracy(
    raw_replacement_results['replacement_logits'], 
    raw_replacement_results['replacement_labels'])
writer.add_scalar('eval/raw_replacement_acc', raw_replacement_acc, 0)
writer.add_scalar('eval/raw_replacement_loss', np.mean(raw_replacement_results['replacement_loss']), 0)

# multi replace results (use prev tok prediction)
tok_probs = softmax(np.array(tok_results['tok_logits']), axis=1)
tok_preds = np.argmax(tok_probs, axis=1) # take token with highest positive prob in each sequence
true_replacement_results = run_inference(replace_model, eval_dataloader, cls_criterion, tok_criterion, tokenizer, tok_preds=tok_preds)
true_replacement_acc = classification_accuracy(
    true_replacement_results['replacement_logits'], 
    true_replacement_results['replacement_labels'],
    prior=tok_hits)
writer.add_scalar('eval/true_replacement_acc', true_replacement_acc, 0)
writer.add_scalar('eval/true_replacement_loss', np.mean(true_replacement_results['replacement_loss']), 0)



# write results
results_file = open(WORKING_DIR + '/results.txt', 'w')

pred_replace_ids = np.argmax(softmax(np.array(true_replacement_results['replacement_logits']), axis=1), axis=1)

for i, batch in enumerate(eval_dataloader):
    input_ids, input_mask, segment_ids, bias_label_ids, tok_label_ids, replace_ids = batch    
    batch_size = eval_dataloader.batch_size

    input_id_seqs = input_ids.cpu().numpy()

    batch_gold_insertions = np.where(tok_label_ids.cpu().numpy() == 1)[1]
    batch_pred_insertions = tok_preds[i * batch_size : (i * batch_size) + batch_size]

    batch_gold_distributions = tok_label_ids.cpu().numpy()
    batch_pred_distributions = tok_probs[i * batch_size : (i * batch_size) + batch_size]

    batch_gold_replacements = replace_ids.cpu().numpy()
    batch_pred_replacements = pred_replace_ids[i * batch_size : (i * batch_size) + batch_size]

    for id_seq, gold_insertion, pred_insertion, gold_dist, pred_dist, gold_replace_id, pred_replace_id in zip(
        input_id_seqs, batch_gold_insertions, batch_pred_insertions,
        batch_gold_distributions, batch_pred_distributions,
        batch_gold_replacements, batch_pred_replacements):

        tok_seq = tokenizer.convert_ids_to_tokens(id_seq)
        if gold_replace_id == num_replacement_labels - 1:
            gold_replace_tok = '<del>'
        else:
            gold_replace_tok = tokenizer.convert_ids_to_tokens([gold_replace_id])[0]
        if pred_replace_id == num_replacement_labels - 1:
            pred_replace_tok = '<del>'
        else:
            pred_replace_tok = tokenizer.convert_ids_to_tokens([pred_replace_id])[0]

        print('#' * 80, file=results_file)
        print('SEQ: \t\t', ' '.join(tok_seq), file=results_file)
        print('GOLD DIST: \t', gold_dist, file=results_file)
        print('PRED DIST: \t', [round(float(x), 2) for x in pred_dist], file=results_file)
        print('GOLD INS: \t', gold_insertion, file=results_file)
        print('PRED INS: \t', pred_insertion, file=results_file)
        print('GOLD TOK: \t', gold_replace_tok.encode('utf-8'), file=results_file)
        print('PRED TOK: \t', pred_replace_tok.encode('utf-8'), file=results_file)

results_file.close()
# print('INITIAL EVAL...')
# model.eval()
# results = run_inference(model, eval_dataloader, cls_criterion, tok_criterion, tokenizer=tokenizer)
# replacement_acc = classification_accuracy(results['replacement_logits'], results['replacement_labels'])
# tok_acc = tag_accuracy(results['tok_logits'], results['bias_labels'], results['tok_labels'], top=1)
# writer.add_scalar('eval/replacement_loss', np.mean(results['replacement_loss']), 0)
# writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
# writer.add_scalar('eval/replacement_acc', replacement_acc, 0)
# writer.add_scalar('eval/tok_acc', tok_acc, 0)

# print('TRAINING...')
# model.train()
# train_step = 0
# for epoch in range(EPOCHS):
#     print('STARTING EPOCH ', epoch)
#     for step, batch in enumerate(train_dataloader):
#         if CUDA:
#             batch = tuple(x.cuda() for x in batch)
#         input_ids, input_mask, segment_ids, bias_label_ids, tok_label_ids, replace_ids = batch

#         bias_indices = np.where(tok_label_ids.cpu().numpy() == 1)[1]
#         replace_logits, tok_logits = model(input_ids, segment_ids, input_mask, labels=bias_indices)

#         replacement_loss = cls_criterion(replace_logits.view(-1, len(tokenizer.vocab)+1), replace_ids.view(-1))
#         tok_loss = tok_criterion(tok_logits.view(-1, NUM_TOK_LABELS), tok_label_ids.view(-1))
        
#         if 'multi' in mode:
#             loss = replacement_loss + tok_loss
#         else:
#             loss = replacement_loss

#         loss.backward()
#         optimizer.step()
#         model.zero_grad()

#         writer.add_scalar('train/replacement_loss', replacement_loss.data[0], train_step)
#         writer.add_scalar('train/tok_loss', tok_loss if isinstance(tok_loss, int) else tok_loss.data[0], train_step)
#         writer.add_scalar('train/loss', loss.data[0], train_step)
#         train_step += 1

#     # eval
#     print('EVAL...')
#     model.eval()
#     results = run_inference(model, eval_dataloader, cls_criterion, tok_criterion, tokenizer=tokenizer)
#     replacement_acc = classification_accuracy(results['replacement_logits'], results['replacement_labels'])
#     tok_acc = tag_accuracy(results['tok_logits'], results['bias_labels'], results['tok_labels'], top=1)
#     writer.add_scalar('eval/replacement_loss', np.mean(results['replacement_loss']), 0)
#     writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
#     writer.add_scalar('eval/replacement_acc', replacement_acc, 0)
#     writer.add_scalar('eval/tok_acc', tok_acc, 0)

#     model.train()



