"""
prepares a corpusfile-style input for mechanical turk input


usage: 
python corpusfile_to_mturk.py [input corpusfile] [pred corpusfile] [output]

e.g.
python parallel_file_to_mturk.py delete_retrieve/inputs.27 delete_retrieve/preds.27 delete_retrieve/delete_retrieve_mturk_input.csv
"""
import sys
import csv
from tqdm import tqdm
from hashlib import md5
import random

pre = sys.argv[1]
pred = sys.argv[2]
out = sys.argv[3]

# TODO ignore soft matching with data stuff. just use
# hashes for ids..

#vectorizer = TfidfVectorizer()
#vectorizer.fit(input_lines)
#key_corpus = vectorizer.transform(input_lines)

#def get_id(query_vec):
#    scores = np.squeeze(np.dot(key_corpus, query_vec.T).toarray())
#    scores_ids = zip(scores, ids, input_lines)
#    selected = sorted(scores_ids, reverse=True)[0]
#    return selected

def detokenize(s):
    out = []
    for w in s.split():
        if w.startswith('##') and len(out) > 0:
            out[-1] += w[2:]
        else:
            out.append(w)
    return ' '.join(out)

with open(out, 'w') as f:
    writer = csv.writer(f)
    n, d = 0, 0
    for l1, l2 in tqdm(zip(open(pre), open(pred))):
        l1 = l1.strip()
        pred = l2.strip()
#        (score, id, pre) = get_id(vectorizer.transform([l1]))

        if random.random() < 0.5:
            writer.writerow([md5(pre).hexdigest(), detokenize(pre), detokenize(pred), '0'])
        else:
            writer.writerow([md5(pre).hexdigest(), detokenize(pred), detokenize(pre), '1'])



