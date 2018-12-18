"""
python make_attribute_vocab.py [vocab] [corpus1] [corpus2] r

subsets a [vocab] file by finding the words most associated with 
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

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


    def salience(self, feature, attribute='pre', lmbda=0.5):
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


vocab = set([w.strip() for i, w in enumerate(open(sys.argv[1]))])
corpus1 = sys.argv[2]
corpus1 = [
    w if w in vocab else '<unk>' 
    for l in open(corpus1)
    for w in l.strip().split()
]

corpus2 = sys.argv[3]
corpus2 = [
    w if w in vocab else '<unk>' 
    for l in open(corpus2)
    for w in l.strip().split()
]

r = float(sys.argv[4])

sc = SalienceCalculator(corpus1, corpus2)

for tok in vocab:
    # todo make this cleaner
    if max(sc.salience(tok, attribute='pre'), sc.salience(tok, attribute='post')) > r:
        print(tok)


