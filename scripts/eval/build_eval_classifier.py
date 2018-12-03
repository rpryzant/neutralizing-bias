# python build_eval_classifier.py ../../data/pre.train ../../data/post.train ../../data/vocab.20000 /Users/rpryzant/persuasion/data/eval_classifier

# builds the classifier needed for "classifier eval" metric

import time

import sys
sys.path.append('../../src/style_transfer_baseline')
import data
import models

corpus1 = sys.argv[1]
corpus2 = sys.argv[2]
vocab_path = sys.argv[3]
save_prefix = sys.argv[4]

vocab, _ = data.build_vocab_maps(vocab_path)

tc = models.TextClassifier(vocab)

print('FITTING...')
start = time.time()
tc.fit(corpus1, corpus2)
print('DONE. TOOK ', time.time() - start)

tc.save(save_prefix)
