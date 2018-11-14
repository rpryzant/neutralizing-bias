import difflib
import sys
"""

['  Gyil', '  repertoire', '  with', '- master', '  xylophonists', '  Bernard', '  Woma', '- ,', '- Jerome', '- Balsab', '- ,', '- and', '- Alfred', '- Kpebsaane', '  .']
1
['master', '  xylophonists', '  Bernard', '  Woma', ',', 'Jerome', 'Balsab', ',', 'and', 'Alfred', 'Kpebsaane']
['  Gyil', '  repertoire', '  with', '  xylophonists', '+ including', '  Bernard', '  Woma', '  .']
3
['including']

['  |', '  demographic', '  =', '! Seinen']
0
[]
['  |', '  demographic', '  =', '! Male']
0
[]

"""

def extract_diff(tok_seq):
    out = []
    island_size = 0
    for tok in tok_seq:
        if tok[0] in '-+!':
            out += [tok[2:]]
            island_size = 0
        else:
            if len(out) > 0:
                out += [tok]
            island_size += 1
    print(island_size)
    return out[:-island_size]

for l in open(sys.argv[1]):
    [_, _, pre, post, _] = l.strip().split('\t')
    print(pre)
    print(post)
    diff = difflib.context_diff(pre.split(), post.split())
    # skip header
    for _ in range(4):
        next(diff)

    diff = [x for x in diff]
    post_start_idx = next( ( (i, x) for i, x in enumerate(diff) if '\n' in x ) )[0]
    
    pre_diff = diff[: post_start_idx]
    post_diff = diff[post_start_idx + 1 :]
    
    print(pre_diff)
    print(extract_diff(pre_diff))

    print(post_diff)
    print(extract_diff(post_diff))


#    print([x for x in difflib.context_diff(pre.split(), post.split())])
    print('#' * 80)

