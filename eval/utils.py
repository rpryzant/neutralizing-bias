import hashlib
from simplediff import diff
import re

def parse_results_file(fp, ignore_unchanged=False):
    """
    args:
        fp: str, pointer to results file
        ignore_unchanged: bool, whether to ignore records where src == pred

    returns:
        {
            id (hash of src str): {
                src: src str
                pred: pred str
                dist: pred dist (list of float)
            }
        }
    """
    def is_complete(d):
        return 'src' in d and 'pred' in d and 'dist' in d

    def punct_diff(a, b):
        d = diff(a.split(), b.split())
        changed_text = ''.join([
            ''.join(chunk).strip() for tag, chunk in d if tag != '='])
        if not re.search('[a-z]', changed_text):
            return True
        elif re.sub(r'[^\w\s]','', a) == re.sub(r'[^\w\s]','', b):
            return True
        return False

    out = {}
    cur = {}
    for l in open(fp):
        l = l.strip()
        if '#########' in l and is_complete(cur):
            src_hash = hashlib.md5(cur['src'].encode()).hexdigest()
            pred_hash = hashlib.md5(cur['pred'].encode()).hexdigest()
            if ignore_unchanged:
                if not src_hash == pred_hash and not punct_diff(str(cur['src']), str(cur['pred'])):
                    out[src_hash] = cur
            else:
                out[src_hash] = cur

            cur = {}
        elif 'IN SEQ' in l:
            # decode into str from bytes
            cur['src'] = eval(l.split('\t')[-1]).decode()
        elif 'PRED SEQ' in l:
            cur['pred'] = eval(l.split('\t')[-1]).decode()
        elif 'PRED DIST' in l:
            cur['dist'] = eval(l.split('\t')[-1])

    return out


if __name__ == '__main__':
    import sys

    # print(parse_results_file(sys.argv[1], True))

    def detokenize(s):
        s = str(s)
        out = []
        for w in s.split():
            if w.startswith('##') and len(out) > 0:
                out[-1] += w[2:]
            else:
                out.append(w)

        return ' '.join(out)

    s = []
    x = parse_results_file(sys.argv[1], True)
    for rec, d in x.items():
        src, pred = detokenize(d['src']), detokenize(d['pred'])
        ratio = float(len(pred)) / len(src)
        if ratio > 1.3:
            print(d['src'])
            print(d['pred'])
            print()
        s.append(ratio)

    import matplotlib.pyplot as plt
    import numpy as np
    
    print(np.mean(s))
    
    n, bins, patches = plt.hist(s, 20, facecolor='blue', alpha=0.5)
    plt.show()