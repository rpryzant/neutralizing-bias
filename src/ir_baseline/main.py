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
from nltk.stem import PorterStemmer

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
        selected = [(self.corpus[i], i, score) for (score, i) in selected]

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





class IRDebiaser(object):
    def __init__(self, corpus_path, attribute_vocab_path=None, stem=False):
        """ if no attribute vocabs, uses pre/post diff
            and assumes all content at test time
        """
        if attribute_vocab_path is not None:
            self.attribute_vocab = set([x.strip() for x in open(attribute_vocab_path)])
        else:
            self.attribute_vocab = None

        self.stemmer = PorterStemmer() if stem else None

        (self.pre_raw, self.post_raw, pre_a, post_a, 
         pre_content, post_content) = self.prep_corpus(corpus_path)

        self.tfidf = TFIDF(post_content)

    def debias(self, s):
        if self.stemmer:
            s = self.stem(s)

        if self.attribute_vocab is not None:
            s_content, _ = self.extract_attributes(
                s.split(), self.attribute_vocab)
        else:
            s_content = s
            
        retrieved_content, idx, score = self.tfidf.most_similar(s, n=10)[0]
        return self.post_raw[idx]


    def prep_corpus(self, corpus_path):
        pre_raw_corpus = []
        post_raw_corpus = []
        pre_content_corpus = []
        post_content_corpus = []
        pre_a_corpus = []
        post_a_corpus = []

        for l in tqdm(open(data_path)):
            [_, _, pre, post, _] = l.strip().split('\t')
            pre = pre.lower()
            post = post.lower()
            pre_raw_corpus.append(pre)
            post_raw_corpus.append(post)

            if self.stemmer:
                pre = self.stem(pre)
                post = self.stem(post)

            if self.attribute_vocab is not None:
                (pre_content, pre_a) = self.extract_attributes(pre.split(), self.attribute_vocab)
                (post_content, post_a) = self.extract_attributes(post.split(), self.attribute_vocab)
            else: 
                content, pre_a, post_a = self.extract_diff(pre, post)
                pre_content, post_content = content[:], content[:]

            pre_content_corpus.append(' '.join(pre_content))
            post_content_corpus.append(' '.join(post_content))
            pre_a_corpus.append(pre_a)
            post_a_corpus.append(post_a)

        return pre_raw_corpus, post_raw_corpus, pre_a_corpus, post_a_corpus, pre_content_corpus, post_content_corpus


    def stem(self, s):
        return ' '.join([self.stemmer.stem(x) for x in s.split()])

    def extract_attributes(self, tok_seq, attribute_vocab):
        content = []
        attribute = []
        for tok in tok_seq:
            if tok in attribute_vocab:
                attribute.append(tok)
            else: 
                content_append(tok)
        return content, attribute


    def extract_diff(self, pre, post):
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

        return content, a_pre, a_post


data_path = sys.argv[1]

db = IRDebiaser(data_path, stem=True)

print(db.debias('retirement'))


