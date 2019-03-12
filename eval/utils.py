import hashlib


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

    out = {}
    cur = {}
    for l in open(fp):
        l = l.strip()
        if '#########' in l and is_complete(cur):
            src_hash = hashlib.md5(cur['src']).hexdigest()
            pred_hash = hashlib.md5(cur['pred']).hexdigest()
            if not (ignore_unchanged and src_hash == pred_hash):
                out[src_hash] = cur
            cur = {}
        elif 'IN SEQ' in l:
            cur['src'] = eval(l.split('\t')[-1])
        elif 'PRED SEQ' in l:
            cur['pred'] = eval(l.split('\t')[-1])
        elif 'PRED DIST' in l:
            cur['dist'] = eval(l.split('\t')[-1])

    return out


if __name__ == '__main__':
    import sys

    print(parse_results_file(sys.argv[1], True))


