# python unique_test_words.py ../../data/pre.train ../../data/post.train ../../data/pre.test ../../data/post.test ../../data/vocab.20000



import sys
sys.path.append('../../src/style_transfer_baseline')
import data

from collections import Counter

src_train = sys.argv[1]
tgt_train = sys.argv[2]
src_test = sys.argv[3]
tgt_test = sys.argv[4]
vocab_path = sys.argv[5]


vocab, _ = data.build_vocab_maps(vocab_path)

src_ctr = Counter(open(src_train).read().split())
tgt_ctr = Counter(open(tgt_train).read().split())

num = 0.0
denom = 0.0
for src, tgt in zip(open(src_test), open(tgt_test)):
	tgt_unique = set(tgt.strip().split()) - set(src.strip().split())
	tgt_unique = [x for x in tgt_unique if x in vocab]
	num += len([x for x in tgt_unique if x in src_ctr or x in tgt_ctr])
	denom += len(tgt_unique)	


print(num)    # 2018.0
print(denom)  # 2018.0