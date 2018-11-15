"""
python main.py ../../data/TEST

"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
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
        self.tfidf_matrix = self.vectorizer.transform(corpus)
        self.s2idx = {s: i for i, s in enumerate(corpus)}

        
    def most_similar(self, s, n=10):
 # use stuff from this:   
#    https://github.com/snap-stanford/crisis/blob/master/reid/src/rankers/tfidf.py
    
        idx = self.s2idx[s]
        
        print(idx)
        quit()



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
tfidf = TFIDF(corpus=pre_c_corpus)
print(time.time() - start)

print()

print(tfidf.corpus_matrix.shape)
print(tfidf2.tfidf.shape)
