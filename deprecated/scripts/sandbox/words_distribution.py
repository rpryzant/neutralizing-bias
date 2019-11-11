"""
distribution of words unique to src or tgt
python words_distribution.py ../../data/pre ../../data/post
"""
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


src_path = sys.argv[1]
tgt_path = sys.argv[2]



def histogram(x, title):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(x)
    ax.set_title(title)
    fig.show()


def src_unique(src, tgt):
    return set(src.strip().split()) - set(tgt.strip().split())

def tgt_unique(src, tgt):
    return set(tgt.strip().split()) - set(src.strip().split())

src_unique_counts, tgt_unique_counts = list(zip(*[
            (len(src_unique(src, tgt)), len(tgt_unique(src, tgt)))
            for src, tgt in zip(open(src_path), open(tgt_path))
]))

plt.hist(src_unique_counts, bins=50, range=(0, 20))
plt.title('number of unique src tokens per sentence')
plt.show()

plt.hist(tgt_unique_counts, bins=50, range=(0, 20))
plt.title('number of unique tgt tokens per sentence')
plt.show()

