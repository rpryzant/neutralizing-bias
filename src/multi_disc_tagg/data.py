import os
import pickle
from tqdm import tqdm
from simplediff import diff
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch


UD_RELATIONS = [
    'PAD', 'acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux',
    'auxpass', 'case', 'cc', 'cc:preconj', 'ccomp', 'compound',
    'compound:prt', 'conj', 'cop', 'csubj', 'csubjpass', 'dep',
    'det', 'det:predet', 'discourse', 'dislocated', 'dobj',
    'expl', 'foreign', 'goeswith', 'iobj', 'list', 'mark', 'mwe',
    'name', 'neg', 'nmod', 'nmod:npmod', 'nmod:poss', 'nmod:tmod',
    'nsubj', 'nsubjpass', 'nummod', 'parataxis', 'punct', 'remnant',
    'reparandum', 'root', 'vocative', 'xcomp', '<UNK>', '<SKIP>'
]
REL2ID = {x: i for i, x in enumerate(UD_RELATIONS)}
# from nltk.data.load('help/tagsets/upenn_tagset.pickle').keys()
POS2ID = {
    'PAD': 0, 'LS': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8, 
    '--': 9, 'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 
    'PRP$': 17, 'WDT': 18, '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 
    'RB': 25, 'RBR': 26, 'RBS': 27, 'VBD': 28, 'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 
    'JJS': 33, 'PDT': 34, 'MD': 35, 'VB': 36, 'WRB': 37, 'NNP': 38, 'EX': 39, 'NNS': 40, 
    'SYM': 41, 'CC': 42, 'CD': 43, 'POS': 44, '<UNK>': 45, '<SKIP>': 46
}



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


def noise_seq(seq, drop_prob=0.25, shuf_dist=3):
    # from https://arxiv.org/pdf/1711.00043.pdf
    def perm(i):
        return i[0] + (shuf_dist + 1) * np.random.random()
    seq = [x for x in seq if np.random.random() > drop_prob]
    seq = [x for _, x in sorted(enumerate(seq), key=perm)]
    return seq


def get_examples(text_path, text_post_path, tok2id, possible_labels, max_seq_len, 
        noise=False, add_del_tok=False, rel_path='', pos_path='', ):
    global REL2ID
    global POS2ID

    label2id = {label: i for i, label in enumerate(possible_labels)}
    label2id['mask'] = len(label2id)
    
    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    skipped = 0 

    out = {
        'pre_ids': [], 'pre_masks': [], 'pre_lens': [], 'post_in_ids': [], 'post_out_ids': [], 
        'tok_label_ids': [], 'replace_ids': [], 'rel_ids': [], 'pos_ids': []
    }

    for i, (line, post_line, line_rels, line_pos) in enumerate(tqdm(
            zip(open(text_path), open(text_post_path), open(rel_path), open(pos_path)))):
        # ignore the unbiased sentences with tagging -- TODO -- toggle this?    
        tokens = line.strip().split() # Pre-tokenized
        post_tokens = post_line.strip().split()
        rels = line_rels.strip().split()
        pos = line_pos.strip().split()

        tok_diff = diff(tokens, post_tokens)
        tok_labels = get_tok_labels(tok_diff)

    
        # make sure everything lines up    
        if len(tokens) != len(tok_labels) or len(tokens) != len(rels) or len(tokens) != len(pos):
            skipped += 1
            continue
        # ignore lines that broke on non-asci chars (TODO FIX THIS)
        if rels.count('<SKIP>') > (len(rels) * 1.0 / 2):
            skipped += 1
            continue

        try:
            replace_token = next( (chunk for tag, chunk in tok_diff if tag == '+') )[0]
            replace_id = tok2id[replace_token]
        except StopIteration:
            # add deletion token into data
            replace_id = tok2id['<del>']    
            if add_del_tok:
                post_tokens.insert(tok_labels.index('1'), '<del>')

        except KeyError:
            skipped += 1
            continue

        # account for [CLS] and [SEP]
        # if len(tokens) >= max_seq_len:
        tokens = tokens[:max_seq_len - 1]
        tok_labels = tok_labels[:max_seq_len - 1]
        post_tokens = post_tokens[:max_seq_len - 1]
        tok_labels = tok_labels[:max_seq_len - 1]
        rels = rels[:max_seq_len - 1]
        pos = pos[:max_seq_len - 1]

        # use cls/sep as start/end...whelp lol
        post_input_tokens = ['行'] + post_tokens
        post_output_tokens = post_tokens + ['止'] 

        try:
            pre_ids = [tok2id[x] for x in tokens]
            if noise:
                pre_ids = noise_seq(pre_ids)
            pre_ids = pad(pre_ids, 0)
            post_in_ids = pad([tok2id[x] for x in post_input_tokens], 0)
            post_out_ids = pad([tok2id[x] for x in post_output_tokens], 0)
            tok_label_ids = pad([label2id[l] for l in tok_labels], 0)
            rel_ids = pad([REL2ID.get(x, REL2ID['<UNK>']) for x in rels], 0)
            pos_ids = pad([POS2ID.get(x, POS2ID['<UNK>']) for x in pos], 0)
            
        except KeyError:
            # TODO FUCK THIS ENCODING BUG!!!
            skipped += 1
            continue

        input_mask = pad([0] * len(tokens), 1)
        pre_len = len(tokens)

        # make sure its a real edit (only if we're not autoencoding)
        if (text_path != text_post_path) and 1 not in tok_label_ids:
            skipped += 1
            continue

        out['pre_ids'].append(pre_ids)
        out['pre_masks'].append(input_mask)
        out['pre_lens'].append(pre_len)
        out['post_in_ids'].append(post_in_ids)
        out['post_out_ids'].append(post_out_ids)
        out['tok_label_ids'].append(tok_label_ids)
        out['replace_ids'].append(replace_id)
        out['rel_ids'].append(rel_ids)
        out['pos_ids'].append(pos_ids)

    print('SKIPPED ', skipped)
    return out


def get_dataloader(data_path, post_data_path, tok2id, batch_size, max_seq_len, 
        pickle_path=None, test=False, noise=False, add_del_tok=False):
    def collate(data):
        # sort by length for packing/padding
        data.sort(key=lambda x: x[2], reverse=True)
        # group by datatype
        [src_id, src_mask, src_len, post_in_id, post_out_id, tok_label, replace_id, rel_ids, pos_ids] = \
            [torch.stack(x) for x in zip(*data)]
        # cut off at max len for unpacking/repadding
        max_len = src_len[0]
        data = [
            src_id[:, :max_len], src_mask[:, :max_len], src_len, 
            post_in_id[:, :max_len+10], post_out_id[:, :max_len+10], 
            tok_label[:, :max_len], replace_id, 
            rel_ids[:, :max_len], pos_ids[:, :max_len]
        ]
        return data

    if pickle_path is not None and os.path.exists(pickle_path):
        train_examples = pickle.load(open(pickle_path, 'rb'))
    else:
        train_examples = get_examples(
            text_path=data_path, 
            text_post_path=post_data_path,
            tok2id=tok2id,
            possible_labels=["0", "1"],
            max_seq_len=max_seq_len,
            noise=noise,
            add_del_tok=add_del_tok,
            rel_path=data_path + '.rel',
            pos_path=data_path + '.pos')

        pickle.dump(train_examples, open(pickle_path, 'wb'))

    train_data = TensorDataset(
        torch.tensor(train_examples['pre_ids'], dtype=torch.long),
        torch.tensor(train_examples['pre_masks'], dtype=torch.uint8), # byte for masked_fill()
        torch.tensor(train_examples['pre_lens'], dtype=torch.long),
        torch.tensor(train_examples['post_in_ids'], dtype=torch.long),
        torch.tensor(train_examples['post_out_ids'], dtype=torch.long),
        torch.tensor(train_examples['tok_label_ids'], dtype=torch.float),  # for masking
        torch.tensor(train_examples['replace_ids'], dtype=torch.long),
        torch.tensor(train_examples['rel_ids'], dtype=torch.long),
        torch.tensor(train_examples['pos_ids'], dtype=torch.long))

    train_dataloader = DataLoader(
        train_data,
        sampler=(SequentialSampler(train_data) if test else RandomSampler(train_data)),
        collate_fn=collate,
        batch_size=batch_size)

    return train_dataloader, len(train_examples['pre_ids'])


