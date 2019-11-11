"""
python make_corpus_vectors.py ../../data/pre.train ../../data/post.train wiki_unigrams.bin ./../../../sent2vec/fasttext


 for vectorizing: https://github.com/epfml/sent2vec
 sent2vec_wiki_unigrams 5GB (600dim, trained on english wikipedia)
 https://drive.google.com/open?id=0B6VhzidiLvjSa19uYWlLUEkzX3c
"""
import os
import sys
import diff_match_patch as dmp_module
import sent2vec # follow pip install instructions on above git page

corpus1 = sys.argv[1]
corpus2 = sys.argv[2]
sent2vec_model = sys.argv[3]
fasttext = sys.argv[4]



def diff(s1, s2):
    dmp = dmp_module.diff_match_patch()
    d = dmp.diff_main(s1, s2)
    dmp.diff_cleanupSemantic(d)
    return d

def set_diff(s1, s2):
	s1 = set(s1.strip().split())
	s2 = set(s2.strip().split())
	return ' '.join(s1 - s2), ' '.join(s2 - s1)
	

tmp1 = open('tmp1', 'w')
tmp2 = open('tmp2', 'w')

model = sent2vec.Sent2vecModel()
model.load_model(sent2vec_model)


for l1, l2 in zip(open(corpus1), open(corpus2)):
	# d = diff(l1.strip(), l2.strip())
	# uni1 = ' '.join([s for idx, s in d if idx == -1] + ['.'])
	# uni2 = ' '.join([s for idx, s in d if idx == 1] + ['.'])

	uni1, uni2 = set_diff(l1, l2)

	if not uni1:
		uni1 = 'empty'
	if not uni2:
		uni2 = 'empty'

	tmp1.write(uni1.strip() + '\n')
	tmp2.write(uni2.strip() + '\n')

	emb1 = model.embed_sentence(uni1.strip())
	emb2 = model.embed_sentence(uni2.strip())

	emb_diff = emb1 - emb2
	
	print(' '.join([str(x) for x in emb_diff[0]]))
quit()

print('Generating vecs for', corpus1)
os.system('./%s print-sentence-vectors %s < tmp1 > corpus1.vecs' % (
	fasttext, sent2vec_model
))

print('Generating vecs for', corpus2)
os.system('./%s print-sentence-vectors %s < tmp2 > corpus2.vecs' % (
	fasttext, sent2vec_model
))
