# python vector_diff.py corpus1.vecs corpus2.vecs > diff.vecs

import sys
import numpy as np

corpus1_vecs = sys.argv[1]
corpus2_vecs = sys.argv[2]

def read_vecs(path):
	out = []
	for l in open(path):
		out.append([float(x) for x in l.strip().split()])
	return np.array(out)

c1v = read_vecs(corpus1_vecs)
c2v = read_vecs(corpus2_vecs)

diff = c1v - c2v

for vec in diff:
	print(' '.join([str(x) for x in vec]))