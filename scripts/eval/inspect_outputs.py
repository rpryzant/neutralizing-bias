""" inspect multiple prediction files
"""

import sys


files = sys.argv[1:]

file_ptrs = [open(fp) for fp in files]

while True:
    lines = [next(fp) for fp in file_ptrs]
    print('\n'.join(s.strip().replace('<s> ', '').replace(' </s>', '') for s in lines))
    print('\n\n')
