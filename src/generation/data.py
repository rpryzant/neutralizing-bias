"""Data utilities."""
import os
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import diff_match_patch as dmp_module
from collections import defaultdict

import torch
from torch.autograd import Variable

from cuda import CUDA
from simplediff import diff
# global dicts for seq info stuff...
# TODO -- do this smarter
INFO2ID = {
    'unbiased': 0,
    'biased': 1,
    '<unk>': 2,
    '<pad>': 3,
    '<s>': 4,
    '</s>': 5,
}
ID2INFO = {
    0: 'unbiased',
    1: 'biased',
    2: '<unk>',
    3: '<pad>',
    4: '<s>',
    5: '</s>',

}


class CorpusSearcher(object):
    def __init__(self, query_corpus, key_corpus, value_corpus, vectorizer):
        self.vectorizer = vectorizer
        self.vectorizer.fit(key_corpus)

        self.query_corpus = query_corpus
        self.key_corpus = key_corpus
        self.value_corpus = value_corpus
        
        # rows = docs, cols = features
        self.key_corpus_matrix = self.vectorizer.transform(key_corpus)

        
    def most_similar(self, key_idx, n=10):
        query = self.query_corpus[key_idx]
        query_vec = self.vectorizer.transform([query])

        scores = np.dot(self.key_corpus_matrix, query_vec.T)
        scores = np.squeeze(scores.toarray())
        scores_indices = zip(scores, range(len(scores)))
        selected = sorted(scores_indices, reverse=True)[:n]
        # use the retrieved i to pick examples from the VALUE corpus
        selected = [
            (query, self.key_corpus[i], self.value_corpus[i], i, score) 
            for (score, i) in selected
        ]

        return selected


def build_vocab_maps(vocab_file):
    assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file
    unk = '<unk>'
    pad = '<pad>'
    sos = '<s>'
    eos = '</s>'

    lines = [x.strip() for x in open(vocab_file)]

    assert lines[0] == unk and lines[1] == pad and lines[2] == sos and lines[3] == eos, \
        "The first words in %s are not %s, %s, %s, %s" % (vocab_file, unk, pad, sos, eos)

    tok_to_id = {}
    id_to_tok = {}
    for i, vi in enumerate(lines):
        tok_to_id[vi] = i
        id_to_tok[i] = vi

    # Extra vocab item for empty attribute lines
    empty_tok_idx =  len(id_to_tok)
    tok_to_id['<empty>'] = empty_tok_idx
    id_to_tok[empty_tok_idx] = '<empty>'

    return tok_to_id, id_to_tok


def extract_attributes(line, attribute_vocab):
    content = []
    attribute = []
    for tok in line:
        if tok in attribute_vocab:
            attribute.append(tok)
        else:
            content.append(tok)
    return content, attribute

def split_with_diff(src_lines, tgt_lines):
    content = []
    src_attr = []
    tgt_attr = []

    for src, tgt in zip(src_lines, tgt_lines):
        sent_diff = diff(src, tgt)
        tok_collector = defaultdict(list)
        for source, chunk in sent_diff:
            tok_collector[source] += chunk

        content.append(tok_collector['='][:])
        src_attr.append(tok_collector['-'][:])
        tgt_attr.append(tok_collector['+'][:])

    return content[:], content[:], src_attr, tgt_attr
        

def get_side_info(src_lines, tgt_lines):
    out = []
    for src, tgt in zip(src_lines, tgt_lines):
        if ' '.join(src) == ' '.join(tgt):
            out.append(['unbiased'])
        else:
            out.append(['biased'])
    return out
    #     n_src_unique = len(set(src) - set(tgt))
    #     n_tgt_unique = len(set(tgt) - set(src))

    #     if n_src_unique == 0 and n_tgt_unique > 0:
    #         out.append( ['insertion'] )
    #     elif n_src_unique > 0 and n_tgt_unique == 0:
    #         out.append( ['deletion'] )
    #     elif set(src) == set(tgt):
    #         out.append( ['unchanged'] )
    #     else:
    #         out.append( ['edit'] )
    # return out


def read_nmt_data(src, config, tgt, train_src=None, train_tgt=None):
    # 1) read data and split into content/attribute
    src_lines = [l.strip().split() for l in open(src, 'r')]
    tgt_lines = [l.strip().split() for l in open(tgt, 'r')]

    use_diff = config['experimental']['use_diff']
    # use attr vocab at test time if use_diff (we don't have the diffs at test time)
    if use_diff and (
            config['experimental']['diff_ignore_test_attribute_rule'] or
            (not train_src and not train_tgt)):
        src_content, tgt_content, src_attribute, tgt_attribute =\
            split_with_diff(src_lines, tgt_lines)
    else:
        attr_vocab = set([
            x.strip() for x in open(config['data']['attribute_vocab'])
        ])
        src_content, src_attribute = list(zip(
            *[extract_attributes(line, attr_vocab) for line in src_lines]
        ))
        tgt_content, tgt_attribute = list(zip(
            *[extract_attributes(line, attr_vocab) for line in tgt_lines]
        ))

    # 2) read in vocab
    src_tok2id, src_id2tok = build_vocab_maps(config['data']['src_vocab'])
    tgt_tok2id, tgt_id2tok = build_vocab_maps(config['data']['tgt_vocab'])

    # 3) build corpus searchers for picking nearby examples
    # train time: just pick attributes that are close to the current (using word distance)
    #  (no test time behavior for source)
    src_dist_measurer = CorpusSearcher(
        query_corpus=[' '.join(x) for x in src_attribute],
        key_corpus=[' '.join(x) for x in src_attribute],
        value_corpus=[' '.join(x) for x in src_attribute],  # fuck you RAM
        vectorizer=CountVectorizer(vocabulary=src_tok2id, binary=True),
    )
    # train time: just pick attributes that are close to the current (using word distance)
    if train_src is None or train_tgt is None:
        tgt_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in tgt_attribute],
            key_corpus=[' '.join(x) for x in tgt_attribute],
            value_corpus=[' '.join(x) for x in tgt_attribute],
            vectorizer=CountVectorizer(vocabulary=tgt_tok2id, binary=True),
        )
    # at test time:
    #   if attribute vocab: 
    #       compare test src content to train tgt contents (using tfidf). retrieve corresponding attributes
    # if diffs: 
    #       use entire test src instead of just content (don't know content/attr at test time)
    else:
        if use_diff:
            assert len(src_lines) == len(src_content)
            query_corpus = [' '.join(x) for x in src_lines]
        else:
            query_corpus = [' '.join(x) for x in src_content]

        tgt_dist_measurer = CorpusSearcher(
            query_corpus=query_corpus,
            key_corpus=[' '.join(x) for x in train_tgt['content']],
            value_corpus=[' '.join(x) for x in train_tgt['attribute']],
            vectorizer=TfidfVectorizer(vocabulary=tgt_tok2id)
        )

    # 4) get some sequence info
    side_info = get_side_info(src_lines, tgt_lines)

    # 5) package errythang up yerp
    src = {
        'data': src_lines, 'content': src_content, 'attribute': src_attribute,
        'tok2id': src_tok2id, 'id2tok': src_id2tok, 'dist_measurer': src_dist_measurer,
        'side_info': side_info
    }
    tgt = {
        'data': tgt_lines, 'content': tgt_content, 'attribute': tgt_attribute,
        'tok2id': tgt_tok2id, 'id2tok': tgt_id2tok, 'dist_measurer': tgt_dist_measurer,
        'side_info': side_info
    }
    return src, tgt


def sample_replace(lines, dist_measurer, sample_rate, corpus_idx):
    """
    replace sample_rate * batch_size lines with nearby examples (according to dist_measurer)
    not exactly the same as the paper (words shared instead of jaccaurd during train) but same idea
    """
    out = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        if random.random() < sample_rate:
            sims = dist_measurer.most_similar(corpus_idx + i)[1:]  # top match is the current line
            try:
                line = next( (
                    tgt_attr.split() for src_cntnt, tgt_cntnt, tgt_attr, _, _ in sims
                    if tgt_attr != ' '.join(line) # and tgt_attr != ''   # TODO -- exclude blanks?
                ) )
            # all the matches are blanks
            except StopIteration:
                line = []
            # TODO: attach start/end backwards and reverse these inputs???
            line = ['<s>'] + line + ['</s>'] 

        # corner case: special tok for empty sequences (just start/end tok)
        if len(line) == 2:
            line.insert(1, '<empty>')
        out[i] = line

    return out


def get_minibatch(lines, tok2id, index, batch_size, max_len, sort=False, idx=None,
        dist_measurer=None, sample_rate=0.0, reverse=False):
    """Prepare minibatch."""
    lines = [
        ['<s>'] + line[:max_len] + ['</s>']
        for line in lines[index:index + batch_size]
    ]
    if dist_measurer is not None:
        lines = sample_replace(lines, dist_measurer, sample_rate, index)

    lens = [len(line) - 1 for line in lines]
    max_len = max(lens)

    unk_id = tok2id['<unk>']
    input_lines = [
        [tok2id.get(w, unk_id) for w in line[:-1]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]
    output_lines = [
        [tok2id.get(w, unk_id) for w in line[1:]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]

    if reverse:
        input_lines = [l[::-1] for l in input_lines]
        output_lines = [l[::-1] for l in output_lines]

    mask = [
        ([1] * l) + ([0] * (max_len - l))
        for l in lens
    ]

    if sort:
        # sort sequence by descending length
        idx = [x[0] for x in sorted(enumerate(lens), key=lambda x: -x[1])]

    if idx is not None:
        lens = [lens[j] for j in idx]
        input_lines = [input_lines[j] for j in idx]
        output_lines = [output_lines[j] for j in idx]
        mask = [mask[j] for j in idx]

    input_lines = Variable(torch.LongTensor(input_lines))
    output_lines = Variable(torch.LongTensor(output_lines))
    mask = Variable(torch.FloatTensor(mask))

    if CUDA:
        input_lines = input_lines.cuda()
        output_lines = output_lines.cuda()
        mask = mask.cuda()

    return input_lines, output_lines, lens, mask, idx


def minibatch(src, tgt, idx, batch_size, max_len, config, is_test=False):
    model_type = config['model']['model_type']
    force_tgt_outputs = config['experimental']['force_tgt_outputs']

    if not is_test:
        use_src = random.random() < 0.5
        in_dataset = src if use_src else tgt
        out_dataset = in_dataset
        attribute_id = 0 if use_src else 1
    else:
        in_dataset = src
        out_dataset = tgt
        attribute_id = 1

    if force_tgt_outputs:
        out_dataset = tgt

    if model_type == 'delete':
        inputs = get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        # get raw source too for evaluation
        raw_src = get_minibatch(
            in_dataset['data'], in_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        # true length could be less than batch_size at edge of data
        batch_len = len(outputs[0])
        attribute_ids = [attribute_id for _ in range(batch_len)]
        attribute_ids = Variable(torch.LongTensor(attribute_ids))
        if CUDA:
            attribute_ids = attribute_ids.cuda()

        attributes = (attribute_ids, None, None, None, None)

    elif model_type == 'delete_retrieve':
        inputs =  get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        attributes =  get_minibatch(
            out_dataset['attribute'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1],
            dist_measurer=out_dataset['dist_measurer'], sample_rate=0.25) # TODO reverse because no packing??
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        # get raw source too for evaluation
        raw_src = get_minibatch(
            in_dataset['data'], in_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

    elif model_type == 'seq2seq':
        # ignore the in/out dataset stuff
        inputs = get_minibatch(
            src['data'], src['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            tgt['data'], tgt['tok2id'], idx, batch_size, max_len, idx=inputs[-1])
        attributes = (None, None, None, None, None)

        raw_src = inputs

    else:
        raise Exception('Unsupported model_type: %s' % model_type)

    # extra side inputs for categorical variables and stuff
    side_info = get_minibatch(
        in_dataset['side_info'], INFO2ID, idx, batch_size, max_len, idx=inputs[-1])

    return inputs, attributes, outputs, side_info, raw_src


def unsort(arr, idx):
    """unsort a list given idx: a list of each element's 'origin' index pre-sorting
    """
    unsorted_arr = arr[:]
    for i, origin in enumerate(idx):
        unsorted_arr[origin] = arr[i]
    return unsorted_arr



