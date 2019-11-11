"""
take a mturk file and recover the raw data examples that are in it

python testset_from_mturk.py trump_sample.csv data/full_test_sets/trump > data/test_set_samples/trump

"""
import sys
import hashlib
import csv
from fuzzywuzzy import process
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


class CorpusSearcher(object):
    def __init__(self, key_corpus, value_corpus, vectorizer):
        self.vectorizer = vectorizer
        self.vectorizer.fit(key_corpus)

        self.key_corpus = key_corpus
        self.value_corpus = value_corpus
        
        # rows = docs, cols = features
        self.key_corpus_matrix = self.vectorizer.transform(key_corpus)

        
    def most_similar(self, query, n=10):
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


mturk_fp = sys.argv[1]
data_fp = sys.argv[2]



#data = {l.strip().split('\t')[3]: l.strip() for l in open(data_fp)}
data = [l.strip().split('\t')[3] for l in open(data_fp)]
lines =  [l.strip() for l in open(data_fp)]
searcher = CorpusSearcher(data, data, TfidfVectorizer())


z = open('out.csv', 'w')
writer = csv.writer(z)

with open(mturk_fp) as f:


    reader = csv.reader(f)
    for row in reader:
        yup = row[1]

        _, match, _, i, _ = searcher.most_similar(yup, n=1)[0]

        #print(match)
        #print(data[i])
# JUST DO THIS IF YOU WANT THE DATAFILE
#        print(lines[i]) 
        revid = lines[i].split('\t')[0]
        
        writer.writerow([revid] + row[1:])


 #       print(data[match])

    
