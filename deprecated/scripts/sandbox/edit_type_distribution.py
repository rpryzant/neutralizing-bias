"""
python edit_type_distribution.py ../../data/pre ../../data/post
"""
import sys




src_path = sys.argv[1]
tgt_path = sys.argv[2]

deletion = 0
insertion = 0
edit = 0

for src, tgt in zip(open(src_path), open(tgt_path)):
    src = set(src.strip().split())
    tgt = set(tgt.strip().split())

    n_src_unique = len(src - tgt)
    n_tgt_unique = len(tgt - src)

    if n_src_unique == 0 and n_tgt_unique > 0:
        insertion += 1
    elif n_src_unique > 0 and n_tgt_unique == 0:
        deletion += 1
    else:
        edit += 1

print('insertions: ', insertion)
print('deletions: ', deletion)
print('edits: ', edit)


