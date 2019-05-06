"""
takes a resultsfile-style output and preps it for
mechanical turk

usage:
python resultsfile_to_mturk.py [resultsfile] [outfile]

"""
import sys
import random
import csv
import hashlib

sys.path.append('../../eval')
import utils # eval/utils.py

in_path = sys.argv[1]
out_path = sys.argv[2]

def detokenize(s):
    s = str(s)
    out = []
    for w in s.split():
        if w.startswith('##') and len(out) > 0:
            out[-1] += w[2:]
        else:
            out.append(w)

    return ' '.join(out)

results = utils.parse_results_file(in_path, ignore_unchanged=False)

with open(out_path, 'w') as f:
    writer = csv.writer(f)
    for id, result_dict in results.items():
        if random.random() < 0.5:
            writer.writerow([
                    id, 
                    detokenize(result_dict['src']), 
                    detokenize(result_dict['pred']), 
                    '0'
                    ])
        else:
            writer.writerow([
                    id, 
                    detokenize(result_dict['pred']), 
                    detokenize(result_dict['src']), 
                    '1'
                    ])



