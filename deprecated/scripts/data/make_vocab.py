"""
args:
1 corpus file (tokenized)
2 K
prints K most frequent vocab items
"""
import sys
from collections import Counter

print('<unk>')
print('<pad>')
print('<s>')
print('</s>')

c = Counter()
for l in open(sys.argv[1]):
    for tok in l.strip().split():
        c[tok] += 1

for tok, _ in c.most_common(int(sys.argv[2])):
    print(tok)



