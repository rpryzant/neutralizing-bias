import pickle
import numpy as np
from simplediff import diff
from sklearn.metrics.pairwise import cosine_similarity


data = pickle.load(open('out.pkl', 'rb'))


revID = data['revID']
src = data['src']
tgt = data['tgt']
src_emb = data['src_emb']
tgt_emb = data['tgt_emb']


diff_vecs = np.array(src_emb) - np.array(tgt_emb)

# print(src[0])
# print(tgt[0])

query = diff_vecs[0]
sims = cosine_similarity([query], diff_vecs[1:]).tolist()[0]
sims_idxs = list(zip(sims, range(len(sims))))
sims_idxs = sorted(sims_idxs)[::-1]

argmax = np.argmax(sims) + 1

print(diff(src[0].split(), tgt[0].split()))

for i in range(10):
	idx = sims_idxs[i][1] + 1
	# idx = sims_idxs[-i][1] + 1
	print(diff(src[idx].split(), tgt[idx].split()))
	print()

# diff_vecs




