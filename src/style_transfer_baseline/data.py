"""Data utilities."""
import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import torch
from torch.autograd import Variable

from cuda import CUDA


class WordDistance(object):
    # TODO -- doesn't work super well on these data...word vectors might be softer/more lenient
    def __init__(self, corpus):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)
        self.corpus = corpus
        # rows = docs, cols = features
        self.tfidf_matrix = self.vectorizer.transform(corpus)
        self.tfidf_matrix = (self.tfidf_matrix != 0).astype(int) # make binary

        
    def most_similar(self, s, n=10):
        assert isinstance(s, str)
        query_tfidf = self.vectorizer.transform([s])
        scores = np.dot(self.tfidf_matrix, query_tfidf.T)
        scores = np.squeeze(scores.toarray())
        scores_indices = zip(scores, range(len(scores)))
        selected = sorted(scores_indices, reverse=True)[:n]
        selected = [(self.corpus[i], i, score) for (score, i) in selected]

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

    return tok_to_id, id_to_tok


def extract_attributes(line, attribute_vocab):
    content = []
    attribute = []
    for tok in line:
        if tok in attribute_vocab:
            attribute.append(tok)
        else:
            content.append(tok)
    return line, content, attribute


def read_nmt_data(src, config, tgt, attribute_vocab):
    # TODO -- handle inference, when there's no tgt
    attribute_vocab = set([x.strip() for x in open(attribute_vocab)])

    src_lines = [l.strip().split() for l in open(src, 'r')]
    src_lines, src_content, src_attribute = list(zip(
        *[extract_attributes(line, attribute_vocab) for line in src_lines]
    ))
    src_attribute_dist = WordDistance([' '.join(x) for x in src_attribute])
    src_tok2id, src_id2tok = build_vocab_maps(config['data']['src_vocab'])
    src = {
        'data': src_lines, 'content': src_content, 'attribute': src_attribute,
        'tok2id': src_tok2id, 'id2tok': src_id2tok, 'attribute_dist': src_attribute_dist
    }

    tgt_lines = [l.strip().split() for l in open(tgt, 'r')] if tgt else None
    tgt_lines, tgt_content, tgt_attribute = list(zip(
        *[extract_attributes(line, attribute_vocab) for line in tgt_lines]
    ))
    tgt_attribute_dist = WordDistance([' '.join(x) for x in tgt_attribute])
    tgt_tok2id, tgt_id2tok = build_vocab_maps(config['data']['tgt_vocab'])
    tgt = {
        'data': tgt_lines, 'content': tgt_content, 'attribute': tgt_attribute,
        'tok2id': tgt_tok2id, 'id2tok': tgt_id2tok, 'attribute_dist': tgt_attribute_dist
    }

    return src, tgt


def get_minibatch(lines, tok2id, index, batch_size, max_len, sort=False, idx=None,
        dist_measurer=None, sample_rate=0.0):
    """Prepare minibatch."""
    lines = [
        ['<s>'] + line[:max_len] + ['</s>']
        for line in lines[index:index + batch_size]
    ]

    if dist_measurer is not None:
        # replace sample_rate * batch_size lines with nearby examples (according to dist_measurer)
        for line in lines:
            if random.random() < sample_rate:
                # take the 2nd closest line in the data (closest = this example)
                line = dist_measurer.most_similar(' '.join(line[1:-1]))[1][0].split()
                line = ['<s>'] + line + ['</s>']


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


def minibatch(src, tgt, idx, batch_size, max_len, model_type):
    # TODO -- TEST TIME!??!
    if random.random() < 0.5:
        dataset = src
        attribute_ids = [0 for _ in range(batch_size)]
    else:
        dataset = tgt
        attribute_ids = [1 for _ in range(batch_size)]

    if model_type == 'delete':
        attribute_ids = Variable(torch.LongTensor(attribute_ids))
        if CUDA:
            attribute_ids = attribute_ids.cuda()

        inputs = get_minibatch(
            dataset['content'], dataset['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            dataset['data'], dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        return inputs, (attribute_ids, None, None, None, None), outputs 

    elif model_type == 'delete_retrieve':

        inputs =  get_minibatch(
            dataset['content'], dataset['tok2id'], idx, batch_size, max_len, sort=True)
#        TODO!!! THE REPLACEMENT THING??
        attributes =  get_minibatch(
            dataset['attribute'], dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1],
            dist_measurer=dataset['attribute_dist'], sample_rate=0.2)
        outputs = get_minibatch(
            dataset['data'], dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        return inputs, attributes, outputs

    else:
        raise Exception('Unsupported model_type: %s' % model_type)






