""" get words with counts > K
"""
from collections import Counter
import sys

counts = Counter(open(sys.argv[1]).read().split())
threshold = int(sys.argv[2])


for tok, count in counts.items():
    if count > threshold:
        print(tok)



