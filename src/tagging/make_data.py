"""
make tagging dataset

python make_data.py ../../data/v2/corpus.biased.raw TEST

"""
import sys
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
import diff_match_patch as dmp_module
from simplediff import diff
from difflib import SequenceMatcher, ndiff



BERT_MODEL = "bert-base-uncased"


corpus = sys.argv[1]
out_dir = sys.argv[2]


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
    s1 = tokenizer.tokenize(s1.strip())
    s2 = tokenizer.tokenize(s2.strip())

    s_diff = diff(s1, s2)
    labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            labels += [0] * len(chunk)
        elif tag == '-':
            labels += [1] * len(chunk)
        else:
            pass
    assert len(labels) == len(s1)
    print(labels)
    print(s1)
    print(s2)
    print(s_diff)
    
    # print(list(ndiff(s1.split(), s2.split())))
    return None, None

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=out_dir + '/cache')

for line in open(corpus):
    line = line.strip()
    pre = line.split('\t')[2]
    post = line.split('\t')[3]
    print(pre)
    print(post)
    pre_tok, labels = tokenize_and_label(pre, post, tokenizer)
    print()