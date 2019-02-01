"""
make tagging dataset

python make_data.py ../../data/v2/corpus.mixed.raw.shuf TEST word

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

mixed_corpus = sys.argv[1]
out_dir = sys.argv[2]
mode = sys.argv[3]
assert mode in ['word', 'token']



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

    word_diff = diff(s1.strip().split(), s2.strip().split())
    one_word_diff = sum([len(chunk) for tag, chunk in word_diff if tag == '-']) == 1

    s1 = tokenizer.tokenize(s1.strip())
    s2 = tokenizer.tokenize(s2.strip())

    s_diff = diff(s1, s2)
    tok_labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            tok_labels += ['0'] * len(chunk)
        elif tag == '-':
            tok_labels += ['1'] * len(chunk)
        else:
            pass
    assert len(tok_labels) == len(s1)


    # decide whether to take this example
    label_sum = sum([int(x) for x in tok_labels])
    if mode == 'word':
        take = one_word_diff or label_sum == 0
    elif mode == 'token':
        take = label_sum <= 1

    bias_label = '0' if label_sum == 0 else '1'

    return s1, tok_labels, bias_label, take


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=out_dir + '/cache')

out_text = []
out_tok_labels = []
out_bias_labels = []
print('TOKENIZING AND LABELING...')
for line in tqdm(open(mixed_corpus)):
    line = line.strip()
    pre = line.split('\t')[2]
    post = line.split('\t')[3]
    pre_tok, tok_labels, bias_label, take = tokenize_and_label(pre, post, tokenizer)

    if take:        
        out_text.append(' '.join(pre_tok))
        out_tok_labels.append(' '.join(tok_labels))
        out_bias_labels.append(bias_label)


# print('SHUFFLING...')
# out = list(zip(out_text, out_tok_labels, out_bias_labels))
# random.shuffle(out) # PRE-SHUFFLED SO WE ALL HAVE THE SAME TEST SET
# [out_text, out_tok_labels] = list(zip(*out))


print('WRITING...')
with open(out_dir + '/text.train', 'w') as f:
    f.write('\n'.join(out_text[2000:]))
with open(out_dir + '/tok_labels.train', 'w') as f:
    f.write('\n'.join(out_tok_labels[2000:]))
with open(out_dir + '/bias_labels.train', 'w') as f:
    f.write('\n'.join(out_bias_labels[2000:]))

with open(out_dir + '/text.test', 'w') as f:
    f.write('\n'.join(out_text[:2000]))
with open(out_dir + '/tok_labels.test', 'w') as f:
    f.write('\n'.join(out_tok_labels[:2000]))
with open(out_dir + '/bias_labels.test', 'w') as f:
    f.write('\n'.join(out_bias_labels[:2000]))





