"""
if gain_wiki_revision_text.py was shareded, join the shards with this
"""
import sys
import os
import pickle
from tqdm import tqdm


root_dir = sys.argv[1]
out_prefix = sys.argv[2]

out = {}

skipped = 0

for fname in tqdm(os.listdir(root_dir)):
    path = os.path.join(root_dir, fname)

    d = pickle.load(open(path, 'rb'))
    
    for k in d:
        if k not in out:
            out[k] = d[k]
        else:
            skipped += 1

print('skipped: ', skipped)
print('out size: ', len(out))

print('writting...')
# https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
bytes_out = pickle.dumps(out)
max_bytes = 2**31 - 1
with open(out_prefix + '.pkl', 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

with open(out_prefix + '.keys', 'w') as f:
    for k in out:
        f.write(k + '\n')
