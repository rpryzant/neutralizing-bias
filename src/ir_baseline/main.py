"""
python main.py ../../data/TEST

"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import difflib
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import math
data_path = sys.argv[1]



class TFIDF(object):

    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)
        self.corpus = corpus
        # rows = docs, cols = features
        self.tfidf_matrix = self.vectorizer.transform(corpus)

        
    def most_similar(self, s, n=10):
        assert isinstance(s, str)
        query_tfidf = self.vectorizer.transform([s])
        scores = np.dot(self.tfidf_matrix, query_tfidf.T)
        scores = np.squeeze(scores.toarray())
        scores_indices = zip(scores, range(len(scores)))
        selected = sorted(scores_indices, reverse=True)[:n]
        selected = [(self.corpus[i], score) for (score, i) in selected]

        return selected


class SalienceCalculator(object):

    def __init__(self, pre_corpus, post_corpus):
        self.vectorizer = CountVectorizer()

        pre_count_matrix = self.vectorizer.fit_transform(pre_corpus)
        self.pre_vocab = self.vectorizer.vocabulary_
        self.pre_counts = np.sum(pre_count_matrix, axis=0)
        self.pre_counts = np.squeeze(np.asarray(self.pre_counts))

        post_count_matrix = self.vectorizer.fit_transform(post_corpus)
        self.post_vocab = self.vectorizer.vocabulary_
        self.post_counts = np.sum(post_count_matrix, axis=0)
        self.post_counts = np.squeeze(np.asarray(self.post_counts))


    def salience(self, feature, attribute='pre', lmbda=1.0):
        assert attribute in ['pre', 'post']

        if feature not in self.pre_vocab:
            pre_count = 0.0
        else:
            pre_count = self.pre_counts[self.pre_vocab[feature]]

        if feature not in self.post_vocab:
            post_count = 0.0
        else:
            post_count = self.post_counts[self.post_vocab[feature]]
        
        if attribute == 'pre':
            return (pre_count + lmbda) / (post_count + lmbda)
        else:
            return (post_count + lmbda) / (pre_count + lmbda)


def extract_diff(pre, post):
    """
    seperates a pre/post edit sentence into
        1) the bits that are shared: content(pre), content(post)   
        2) the bits that were changed: attribute(pre), attribute(post)
    """
    dif_toks = [x for x in difflib.ndiff(pre.split(), post.split())]

    content = []
    a_pre = []
    a_post = []
    
    cur = None
    for tok in dif_toks:
        if tok.startswith('-'):
            a_pre.append( tok[2:] )
        elif tok.startswith('+'):
            a_post.append( tok[2:] )
        elif tok.startswith('?'):
            continue
        else:
            content.append( tok.strip() )

    return content[:], a_pre, content[:], a_post



pre_raw_corpus = []
pre_c_corpus = []
pre_a_corpus = []

post_raw_corpus = []
post_c_corpus = []
post_a_corpus = []


for l in tqdm(open(data_path)):
    [_, _, pre, post, _] = l.strip().split('\t')
    
    pre_c, pre_a, post_c, post_a = extract_diff(pre, post)
    
    pre_raw_corpus.append(pre)
    pre_c_corpus.append(' '.join(pre_c).lower())
    pre_a_corpus.append(' '.join(pre_a))
    
    post_raw_corpus.append(post)
    post_c_corpus.append(post_c)
    post_a_corpus.append(post_a)

import time

start = time.time()
sc = SalienceCalculator(pre_raw_corpus, post_raw_corpus)

for x in sc.pre_vocab:
    start = time.time()
    s = sc.salience(x, attribute='pre')
#    print(x, s)
    if s > 5.0:
        print(x, s)

#start = time.time()
#tfidf = TFIDF(corpus=pre_c_corpus)
#print(time.time() - start)

#start = time.time()
#print(tfidf.most_similar('and then i was in a bar and so drunk yeesh', n=10))
#print(time.time() - start)

