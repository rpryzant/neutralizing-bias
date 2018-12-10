"""
make tagging dataset

python make_data.py ../../data/v2/corpus.biased.raw.shuf TEST

"""
import sys
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
import diff_match_patch as dmp_module
from simplediff import diff
from difflib import SequenceMatcher, ndiff
import random
from tqdm import tqdm

BERT_MODEL = "bert-base-uncased"


random.seed(420)

corpus = sys.argv[1]
out_dir = sys.argv[2]




def diff_wordMode(s1, s2):
    #https://github.com/google/diff-match-patch/wiki/Line-or-Word-Diffs
    dmp = dmp_module.diff_match_patch()
    a = dmp.diff_linesToChars(s1, s2)
    print(a); quit()




def diff2_DROPPED(s1, s2):
    sm = SequenceMatcher(None, s1, s2)
    out = []
    for tag, alo, ahi, blo, bhi in sm.get_opcodes():
        if tag == 'replace':
            out.append(('-', s1[alo:ahi]))
            out.append(('+', s2[blo:bhi]))
        elif tag == 'delete':
            out.append(('-', s1[alo:ahi]))
        elif tag == 'insert':
            out.append(('+', s2[blo:bhi]))
        elif tag == 'equal':
            out.append(('=', s1[alo:ahi]))
    return out

def diff3_DROPPED(s1, s2):
    dmp = dmp_module.diff_match_patch()
    d = dmp.diff_main(s1, s2)
    dmp.diff_cleanupSemantic(d)
    return d


def tokenize_and_label(s1, s2, tokenizer):
    s_diff_tmp = diff(s1.strip().split(), s2.strip().split())
    take = sum([len(chunk) for tag, chunk in s_diff_tmp if tag == '-']) == 1

    s1 = tokenizer.tokenize(s1.strip())
    s2 = tokenizer.tokenize(s2.strip())

    s_diff = diff(s1, s2)
    labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            labels += ['0'] * len(chunk)
        elif tag == '-':
            labels += ['1'] * len(chunk)
        else:
            pass
    assert len(labels) == len(s1)

    return s1, labels, take #True


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=out_dir + '/cache')

out_text = []
out_labels = []
print('TOKENIZING AND LABELING...')
for line in tqdm(open(corpus)):
    line = line.strip()
    pre = line.split('\t')[2]
    post = line.split('\t')[3]
    pre_tok, labels, take = tokenize_and_label(pre, post, tokenizer)

    # TODO -- follow http://www.aclweb.org/anthology/P13-1162 and only take single-word changes
    if sum([int(x) for x in labels]) == 1: #take
        out_text.append(' '.join(pre_tok))
        out_labels.append(' '.join(labels))

print('SHUFFLING...')
out = list(zip(out_text, out_labels))
# random.shuffle(out) # PRE-SHUFFLED SO WE ALL HAVE THE SAME TEST SET
[out_text, out_labels] = list(zip(*out))



print('WRITING...')
with open(out_dir + '/text.train', 'w') as f:
    f.write('\n'.join(out_text[1000:]))
with open(out_dir + '/labels.train', 'w') as f:
    f.write('\n'.join(out_labels[1000:]))
with open(out_dir + '/text.test', 'w') as f:
    f.write('\n'.join(out_text[:1000]))
with open(out_dir + '/labels.test', 'w') as f:
    f.write('\n'.join(out_labels[:1000]))





