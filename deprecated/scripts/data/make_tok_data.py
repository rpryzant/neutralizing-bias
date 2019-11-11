"""
subset word data data to only include cases where a single token was 
deleted or replaced

python test.py ../raw/tst.biased > ../raw/tst.tokbiased
"""

import sys

from simplediff import diff

i = 0
for l in open(sys.argv[1]):
    parts = l.strip().split('\t')

    if len(parts) != 7:
        continue

    pre_tok = parts[1].split()
    post_tok = parts[2].split()
    d =  diff(pre_tok, post_tok)

    old = [x for x in d if x[0] == '-']
    new = [x for x in d if x[0] == '+']

#    print(diff(l1.strip().split(), l2.strip().split()))
    if len(old) == 1 and len(old[0][1]) == 1:
        if not new:
            print(l.strip())
            i += 1
        if len(new) == 1 and len(new[0][1]) == 1:
            print(l.strip())
            i += 1


