# python seq2seq.py --train ../../data/v4/tok/biased --test ../../data/v4/tok/biased --working_dir TEST/

import argparse
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


import seq2seq_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    help="train prefix",
    required=True
)
parser.add_argument(
    "--test",
    help="test prefix",
    required=True
)
parser.add_argument(
    "--mode",
    help="model type",
    type=str
)
parser.add_argument(
    "--working_dir",
    help="train continuously on one batch of data",
    type=str
)
parser.add_argument(
    "--attn_type",
    help="bilinear, bert",
    type=str, default=''
)
parser.add_argument(
    "--attn_dropout",
    help="use dropout on bert attn",
    action='store_true'
)
parser.add_argument(
    "--compress_heads",
    help="compress bert attention heads with matrix",
    action='store_true'
)

args = parser.parse_args()


BERT_MODEL = "bert-base-uncased"

train_data_prefix = args.train
test_data_prefix = args.test
mode = args.mode
working_dir = args.working_dir
if not os.path.exists(working_dir):
    os.makedirs(working_dir)


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

TRAIN_BATCH_SIZE = 3
TEST_BATCH_SIZE = 16

EPOCHS = 2

MAX_SEQ_LEN = 5#100

CUDA = (torch.cuda.device_count() > 0)


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

    tok2id = tokenizer.vocab
    tok2id['<del>'] = len(tok2id)
    
    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    skipped = 0 

    out = {
        'pre_ids': [], 'pre_masks': [], 'pre_lens': [], 'post_in_ids': [], 'post_out_ids': [], 
        'tok_label_ids': [], 'replace_ids': []
    }

    for i, (line, post_line, tok_labels, bias_label) in enumerate(tqdm(zip(open(text_path), open(text_post_path), open(tok_labels_path), open(bias_labels_path)))):
        # ignore the unbiased sentences with tagging -- TODO -- toggle this?    
        tokens = line.strip().split() # Pre-tokenized
        post_tokens = post_line.strip().split()
        tok_diff = diff(tokens, post_tokens)
        tok_labels = get_tok_labels(tok_diff)
        assert len(tokens) == len(tok_labels)

        try:
            replace_token = next( (chunk for tag, chunk in tok_diff if tag == '+') )[0]
            replace_id = tok2id[replace_token]
        except StopIteration:
            # add deletion token into data
            post_tokens.insert(tok_labels.index('1'), '<del>')
            replace_id = tok2id['<del>']
        except KeyError:
            skipped += 1
            continue

        # account for [CLS] and [SEP]
        if len(tokens) >= max_seq_len:
            tokens = tokens[:max_seq_len - 1]
            tok_labels = tok_labels[:max_seq_len - 1]
            post_tokens = post_tokens[:max_seq_len - 1]
            tok_labels = tok_labels[:max_seq_len - 1]

        # use cls/sep as start/end...whelp lol
        post_input_tokens = ['[CLS]'] + post_tokens
        post_output_tokens = post_tokens + ['[SEP]']

        try:
            pre_ids = pad([tok2id[x] for x in tokens], 0)
            post_in_ids = pad([tok2id[x] for x in post_input_tokens], 0)
            post_out_ids = pad([tok2id[x] for x in post_output_tokens], 0)
            tok_label_ids = pad([label2id[l] for l in tok_labels], 0)
        except KeyError:
            # TODO FUCK THIS ENCODING BUG!!!
            skipped += 1
            continue

        input_mask = pad([0] * len(tokens), 1)
        pre_len = len(tokens)

        if 1 not in tok_label_ids: # make sure its a real edit
            skipped += 1
            continue

        out['pre_ids'].append(pre_ids)
        out['pre_masks'].append(input_mask)
        out['pre_lens'].append(pre_len)
        out['post_in_ids'].append(post_in_ids)
        out['post_out_ids'].append(post_out_ids)
        out['tok_label_ids'].append(tok_label_ids)
        out['replace_ids'].append(replace_id)

    print('SKIPPED ', skipped)
    return out, tok2id


def get_dataloader(data_path, post_data_path, tok_labels_path, bias_labels_path, tokenizer, batch_size, pickle_path=None, test=False):
    def collate(data):
        # sort by length for packing/padding
        data.sort(key=lambda x: x[2], reverse=True)
        # group by datatype
        [src_id, src_mask, src_len, post_in_id, post_out_id, tok_label, replace_id] = [torch.stack(x) for x in zip(*data)]
        # cut off at max len for unpacking/repadding
        max_len = src_len[0]
        data = [
            src_id[:, :max_len], src_mask[:, :max_len], src_len, 
            post_in_id[:, :max_len+5], post_out_id[:, :max_len+5], 
            tok_label[:, :max_len], replace_id
        ]
        return data

    if pickle_path is not None and os.path.exists(pickle_path):
        train_examples, tok2id = pickle.load(open(pickle_path, 'rb'))
    else:
        train_examples, tok2id = get_examples(
            text_path=data_path, 
            text_post_path=post_data_path,
            tok_labels_path=tok_labels_path,
            bias_labels_path=bias_labels_path,
            tokenizer=tokenizer,
            possible_labels=["0", "1"],
            max_seq_len=MAX_SEQ_LEN)
        pickle.dump((train_examples, tok2id), open(pickle_path, 'wb'))

    train_data = TensorDataset(
        torch.tensor(train_examples['pre_ids'], dtype=torch.long),
        torch.tensor(train_examples['pre_masks'], dtype=torch.uint8), # byte for masked_fill()
        torch.tensor(train_examples['pre_lens'], dtype=torch.long),
        torch.tensor(train_examples['post_in_ids'], dtype=torch.long),
        torch.tensor(train_examples['post_out_ids'], dtype=torch.long),
        torch.tensor(train_examples['tok_label_ids'], dtype=torch.long),
        torch.tensor(train_examples['replace_ids'], dtype=torch.long))

    train_dataloader = DataLoader(
        train_data,
        sampler=(SequentialSampler(train_data) if test else RandomSampler(train_data)),
        collate_fn=collate,
        batch_size=batch_size)

    return train_dataloader, len(train_examples['pre_ids']), tok2id

def train(model, dataloader, epochs, writer, tok2id):
    global CUDA

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    weight_mask = torch.ones(len(tok2id))
    weight_mask[0] = 0
    criterion = nn.CrossEntropyLoss(weight=weight_mask)

    if CUDA:
        weight_mask = weight_mask.cuda()
        loss_criterion = loss_criterion.cuda()

    model.train()
    train_step = 0
    for epoch in range(epochs):
        print('STARTING EPOCH ', epoch)
        for step, batch in enumerate(train_dataloader):
            while True:
                if CUDA:
                    batch = tuple(x.cuda() for x in batch)
                pre_id, pre_mask, pre_len, post_in_id, post_out_id, tok_label_id, replace_id = batch

                post_logits, post_probs = model(pre_id, post_in_id, pre_mask, pre_len)
                loss = criterion(post_logits.contiguous().view(-1, len(tok2id)), post_out_id.contiguous().view(-1))
                print(pre_id)
                print(post_in_id)
                print(post_out_id)                
                print(post_probs.detach().numpy().argmax(axis=-1))
                print(loss)

                loss.backward()
                optimizer.step()
                model.zero_grad()

                writer.add_scalar('train/loss', loss.data[0], train_step)
                train_step += 1
    



tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')

train_dataloader, num_train_examples, tok2id = get_dataloader(
    TRAIN_TEXT, TRAIN_TEXT_POST, TRAIN_TOK_LABELS, TRAIN_BIAS_LABELS, 
    tokenizer, TRAIN_BATCH_SIZE, WORKING_DIR + '/train_data.pkl')
eval_dataloader, num_eval_examples, _ = get_dataloader(
    TEST_TEXT, TEST_TEXT_POST, TEST_TOK_LABELS, TEST_BIAS_LABELS,
    tokenizer, TEST_BATCH_SIZE, WORKING_DIR + '/test_data.pkl',
    test=True)

model = seq2seq_model.Seq2Seq(
    vocab_size=len(tok2id),
    hidden_size=64,#256,
    emb_dim=64,#256,
    dropout=0.2)

writer = SummaryWriter(WORKING_DIR)
train(model, train_dataloader, 1, writer, tok2id)





































