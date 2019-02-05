import os
import pickle
from tqdm import tqdm
from simplediff import diff
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from random import shuffle
import random

import sys; sys.path.append('.')
from shared.args import ARGS


RELATIONS = [
    'det', 'amod', 'nsubj', 'prep', 'pobj', 'ROOT', 
    'attr', 'punct', 'advmod', 'compound', 'acl', 'agent', 
    'aux', 'ccomp', 'dobj', 'cc', 'conj', 'appos', 'nsubjpass', 
    'auxpass', 'poss', 'nummod', 'nmod', 'relcl', 'mark', 
    'advcl', 'pcomp', 'npadvmod', 'preconj', 'neg', 'xcomp', 
    'csubj', 'prt', 'parataxis', 'expl', 'case', 'acomp', 'predet',
    'quantmod', 'dep', 'oprd', 'intj', 'dative', 'meta', 'csubjpass', 
    '<UNK>'
]
REL2ID = {x: i for i, x in enumerate(RELATIONS)}
# from nltk.data.load('help/tagsets/upenn_tagset.pickle').keys()
POS_TAGS = [
    'DET', 'ADJ', 'NOUN', 'ADP', 'NUM', 'VERB', 'PUNCT', 'ADV', 
    'PART', 'CCONJ', 'PRON', 'X', 'INTJ', 'PROPN', 'SYM',
    '<UNK>'
]
POS2ID = {x: i for i, x in enumerate(POS_TAGS)}

# 0: deletion, 1: edit
EDIT_TYPE2ID = {'0': 0, '1': 1}

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def get_tok_labels(s_diff):
    pre_tok_labels = []
    post_tok_labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            pre_tok_labels += [0] * len(chunk)
            post_tok_labels += [0] * len(chunk)
        elif tag == '-':
            pre_tok_labels += [1] * len(chunk)
        elif tag == '+':
            post_tok_labels += [1] * len(chunk)
        else:
            pass

    return pre_tok_labels, post_tok_labels


def noise_seq(seq, drop_prob=0.25, shuf_dist=3, drop_set=None, keep_bigrams=False):
    # from https://arxiv.org/pdf/1711.00043.pdf
    def perm(i):
        return i[0] + (shuf_dist + 1) * np.random.random()
    
    if drop_set == None:
        dropped_seq = [x for x in seq if np.random.random() > drop_prob]
    else:
        dropped_seq = [x for x in seq if not (x in drop_set and np.random.random() < drop_prob)]

    if keep_bigrams:
        i = 0
        original = ' '.join(seq)
        tmp = []
        while i < len(dropped_seq)-1:
            if ' '.join(dropped_seq[i : i+2]) in original:
                tmp.append(dropped_seq[i : i+2])
                i += 2
            else:
                tmp.append([dropped_seq[i]])
                i += 1

        dropped_seq = tmp

    # global shuffle
    if shuf_dist == -1:
        shuffle(dropped_seq)
    # local shuffle
    elif shuf_dist > 0:
        dropped_seq = [x for _, x in sorted(enumerate(dropped_seq), key=perm)]
    # shuf_dist of 0 = no shuffle

    if keep_bigrams:
        dropped_seq = [z for y in dropped_seq for z in y]
    
    return dropped_seq


def get_examples(data_path, tok2id, possible_labels, max_seq_len, 
                 noise=False, add_del_tok=False,
                 tok_dist_path=None, categories_path=None):
    global REL2ID
    global POS2ID
    global ARGS

    label2id = {label: i for i, label in enumerate(possible_labels)}
    label2id['mask'] = len(label2id)

    if ARGS.drop_words is not None:
        drop_set = set([l.strip() for l in open(ARGS.drop_words)])
    else:
        drop_set = None

    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    skipped = 0 

    out = {
        'pre_ids': [], 'pre_masks': [], 'pre_lens': [], 
        'post_in_ids': [], 'post_out_ids': [], 
        'pre_tok_label_ids': [], 'post_tok_label_ids': [], 'tok_dists': [],
        'replace_ids': [], 'rel_ids': [], 'pos_ids': [],
        'type_ids': [], 'categories': []
    }        

    if categories_path is not None:
        category_fp = open(categories_path)

    if tok_dist_path is not None:
        tok_dist_fp = open(tok_dist_path)

    for i, (line) in enumerate(tqdm(open(data_path))):
        parts = line.strip().split('\t')
        if len(parts) == 7:
            [revid, pre, post, _, _, pos, rels] = parts
        elif len(parts) == 5:
            [revid, pre, post, _, _] = parts
            pos = ' '.join(['<UNK>'] * len(pre.strip().split()))
            rels = ' '.join(['<UNK>'] * len(pre.strip().split()))
        else:
            skipped += 1
            continue

        # ignore the unbiased sentences with tagging -- TODO -- toggle this?    
        tokens = pre.strip().split() # Pre-tokenized
        post_tokens = post.strip().split()
        rels = rels.strip().split()
        pos = pos.strip().split()

        tok_diff = diff(tokens, post_tokens)
        pre_tok_labels, post_tok_labels = get_tok_labels(tok_diff)

        if tok_dist_path is not None:
            line = next(tok_dist_fp)
            tok_dist = [float(x) for x in line.strip().split()]

            if ARGS.tok_dist_argmax:
                tmp = [0.0] * len(tokens)
                maxi = max(zip(tok_dist, range(len(tok_dist))))[1]
                tmp[maxi] = 1
                while maxi < len(tokens) and tokens[maxi].startswith('##'):
                    tmp[maxi] = 1.0
                    maxi += 1
                tok_dist = tmp

            if ARGS.tok_dist_threshold > 0.0:
                tmp = [0.0] * len(tokens)
                seed_indices = [i for i, x in enumerate(tok_dist) if x > ARGS.tok_dist_threshold]
                for idx in seed_indices:
                    tmp[idx] = 1.0
                    while idx < len(tokens) and tokens[idx].startswith('##'):
                        tmp[idx] = 1.0
                        idx += 1

                tok_dist = tmp
            
        else:
            tok_dist = pre_tok_labels

                        
        if categories_path is not None:
            line = next(category_fp)
            categories = np.array([float(x) for x in line.strip().split()])
        else:
            categories = np.random.uniform(size=43)   # number of categories
            categories = categories / sum(categories) # normalize
        if ARGS.argmax_categories:
            argmax = np.argmax(categories)
            categories = np.zeros_like(categories)
            categories[argmax] = 1.0
        if ARGS.category_threshold > 0.0:
            categories[categories < ARGS.category_threshold] = 0.0
            categories[categories > ARGS.category_threshold] = 1.0


        # make sure everything lines up    
        if len(tokens) != len(pre_tok_labels) \
            or len(tokens) != len(rels) \
            or len(tokens) != len(pos) \
            or len(post_tokens) != len(post_tok_labels) \
            or len(pre_tok_labels) != len(tok_dist):
            skipped += 1
            continue

        try:
            replace_token = next( (chunk for tag, chunk in tok_diff if tag == '+') )[0]
            replace_id = tok2id[replace_token]
        except StopIteration:
            # add deletion token into data
            replace_id = tok2id['<del>']
            if add_del_tok:
                post_tokens.insert(pre_tok_labels.index(1), '<del>')

        except KeyError:
            skipped += 1
            continue

        # leave room in the post for start/stop and possible category/class token
        if len(tokens) > max_seq_len - 1 or len(post_tokens) > max_seq_len - 1:
            skipped += 1
            continue

        if ARGS.predict_categories:
            tokens = ['[CLS]'] + tokens
            pre_tok_labels = [label2id['mask']] + pre_tok_labels
            post_tok_labels = [label2id['mask']] + post_tok_labels
        elif ARGS.category_input:
            category_id = np.argmax(categories)
            tokens = ['[unused%d]' % category_id] + tokens
            pre_tok_labels = [label2id['mask']] + pre_tok_labels
            post_tok_labels = [label2id['mask']] + post_tok_labels


        post_input_tokens = ['行'] + post_tokens
        post_output_tokens = post_tokens + ['止'] 

        try:
            if noise:
                pre_toks = noise_seq(
                    tokens[:], 
                    drop_prob=ARGS.noise_prob, 
                    shuf_dist=ARGS.shuf_dist,
                    drop_set=drop_set,
                    keep_bigrams=ARGS.keep_bigrams)
            else:
                pre_toks = tokens
            pre_ids = [tok2id[x] for x in pre_toks]
            pre_ids = pad(pre_ids, 0)
            post_in_ids = pad([tok2id[x] for x in post_input_tokens], 0)
            post_out_ids = pad([tok2id[x] for x in post_output_tokens], 0)
            pre_tok_label_ids = pad(pre_tok_labels, label2id['mask'])
            post_tok_label_ids = pad(post_tok_labels, label2id['mask'])
            tok_dist = pad(tok_dist, 0)
            rel_ids = pad([REL2ID.get(x, REL2ID['<UNK>']) for x in rels], 0)
            pos_ids = pad([POS2ID.get(x, POS2ID['<UNK>']) for x in pos], 0)
        except KeyError:
            # TODO FUCK THIS ENCODING BUG!!!
            skipped += 1
            continue

        input_mask = pad([0] * len(tokens), 1)
        pre_len = len(tokens)

        out['type_ids'].append( 0 if sum(post_tok_label_ids) == 0 else 1 )
        out['pre_ids'].append(pre_ids)
        out['pre_masks'].append(input_mask)
        out['pre_lens'].append(pre_len)
        out['post_in_ids'].append(post_in_ids)
        out['post_out_ids'].append(post_out_ids)
        out['pre_tok_label_ids'].append(pre_tok_label_ids)
        out['post_tok_label_ids'].append(post_tok_label_ids)
        out['tok_dists'].append(tok_dist)
        out['replace_ids'].append(replace_id)
        out['rel_ids'].append(rel_ids)
        out['pos_ids'].append(pos_ids)
        out['categories'].append(categories)

    print('SKIPPED ', skipped)
    return out


def get_dataloader(data_path, tok2id, batch_size, max_seq_len, 
                   pickle_path=None, test=False, noise=False, add_del_tok=False, 
                   tok_dist_path=None, categories_path=None, sort_batch=True):
    global ARGS

    def collate(data):
        if sort_batch:
            # sort by length for packing/padding
            data.sort(key=lambda x: x[2], reverse=True)
        # group by datatype
        [
            src_id, src_mask, src_len, 
            post_in_id, post_out_id, 
            pre_tok_label, post_tok_label, tok_dist,
            replace_id, rel_ids, pos_ids,
            type_ids, categories
        ] = [torch.stack(x) for x in zip(*data)]
        # cut off at max len for unpacking/repadding
        max_len = max(src_len)


        # tok_dist options!
        pre_tok_label = pre_tok_label[:, :max_len]
        tok_dist = tok_dist[:, :max_len]
        
        # not noise is a proxy for avoiding pretrain
        if not test and not noise:
            if random.random() < ARGS.tok_dist_softmax_prob and not test:
                tok_dist = torch.nn.functional.gumbel_softmax(tok_dist, tau=0.5)

            if random.random() < ARGS.tok_dist_mix_prob:
                tok_dist = pre_tok_label.clone()
                tok_dist[tok_dist == 2] = 0

            if random.random() < ARGS.tok_dist_noise_prob:
                tok_dist = torch.zeros_like(tok_dist)
                for seq, seq_len in zip(tok_dist, src_len):
                    seq[int(random.random() * int(seq_len))] = 1.0
        # end tok dist options

        data = [
            src_id[:, :max_len], src_mask[:, :max_len], src_len, 
            post_in_id[:, :max_len+10], post_out_id[:, :max_len+10],    # +10 for wiggle room
            pre_tok_label, post_tok_label[:, :max_len+10], tok_dist, # +10 for post_toks_labels too (it's just gonna be matched up with post ids)
            replace_id, rel_ids[:, :max_len], pos_ids[:, :max_len],
            type_ids, categories
        ]
        return data

    if pickle_path is not None and os.path.exists(pickle_path):
        examples = pickle.load(open(pickle_path, 'rb'))
    else:
        examples = get_examples(
            data_path=data_path, 
            tok2id=tok2id,
            possible_labels=["0", "1"],
            max_seq_len=max_seq_len,
            noise=noise,
            add_del_tok=add_del_tok,
            tok_dist_path=tok_dist_path,
            categories_path=categories_path)

        pickle.dump(examples, open(pickle_path, 'wb'))

    data = TensorDataset(
        torch.tensor(examples['pre_ids'], dtype=torch.long),
        torch.tensor(examples['pre_masks'], dtype=torch.uint8), # byte for masked_fill()
        torch.tensor(examples['pre_lens'], dtype=torch.long),
        torch.tensor(examples['post_in_ids'], dtype=torch.long),
        torch.tensor(examples['post_out_ids'], dtype=torch.long),
        torch.tensor(examples['pre_tok_label_ids'], dtype=torch.float),  # for compartin to enrichment stuff
        torch.tensor(examples['post_tok_label_ids'], dtype=torch.float),  # for loss multiplying
        torch.tensor(examples['tok_dists'], dtype=torch.float),   # for enrichment, same as pre_tok_label_ids if the arg isn't set
        torch.tensor(examples['replace_ids'], dtype=torch.long),
        torch.tensor(examples['rel_ids'], dtype=torch.long),
        torch.tensor(examples['pos_ids'], dtype=torch.long),
        torch.tensor(examples['type_ids'], dtype=torch.float),
        torch.tensor(examples['categories'], dtype=torch.float))

    dataloader = DataLoader(
        data,
        sampler=(SequentialSampler(data) if test else RandomSampler(data)),
        collate_fn=collate,
        batch_size=batch_size)

    return dataloader, len(examples['pre_ids'])


