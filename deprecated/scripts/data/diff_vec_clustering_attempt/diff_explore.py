import sys
import numpy as np
from sklearn.cluster import DBSCAN
import time

import matplotlib.pyplot as plt


def read_vecs(path):
	out = []
	for l in open(path):
		out.append([float(x) for x in l.strip().split()])
	return np.array(out)


difs = read_vecs(sys.argv[1])

mu = np.mean(difs)
sd = np.std(difs)
print(mu)
print(sd)
eps = sd * 0.01
min_samples = 4 #? total guess...do better?

print('dbscanning...')
start = time.time()
clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=4).fit(difs)
print('done! took ', time.time() - start)

labels = list(clustering.labels_)

print(labels)
plt.hist(labels)
plt.title('labels')
plt.show()



