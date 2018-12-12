import re
import sys
import csv
import operator
import numpy as np
import string, pickle, os
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
from tqdm import tqdm
from util import *

import mwparserfromhell

in_file = sys.argv[1]
out_file = sys.argv[2]



printable = set(string.printable)

csv.field_size_limit(sys.maxsize)

# special characters
separator = 0
mask_char = 1 
unknown   = 2
to_TBD    = 3
offset    = 4

def wiki_text_clean(text):
    # text_ = filter(lambda x: x in printable, text)
    text_ = ''.join(filter(lambda x:x in string.printable, text)).encode('utf-8')
    # tokens = wordpunct_tokenize(text_)
    # tokens = [s.encode('utf-8') for s in tokens]
    return text_

def collect_revision_text(rev_ids):
    rev_size = len(rev_ids)
    out = {}

    for rev_id in tqdm(rev_ids):
        print('processing revision id = ' + str(rev_id))

        url = 'https://en.wikipedia.org/wiki/?diff=' + str(rev_id)
        prevs_, nexts_ = url2diff(url)

        assert len(prevs_) == len(nexts_), 'corpus sizes not equal!'

        prevs, nexts = [], []

        for pre, post in zip(prevs_, nexts_):
            prevs.append( wiki_text_clean(pre) )
            nexts.append( wiki_text_clean(post) )

        if len(prevs) > 0 and len(nexts) > 0:
            out[rev_id] = (prevs, nexts)

    return out


def go(filename):
    # rev_id    rev_comment    rev_timestamp
    with open(filename, 'r') as f:
        data = list(csv.reader(f, delimiter='\t'))

    rev_ids = [r[0] for r in data]

    X = collect_revision_text(rev_ids)

    pickle.dump(X, open(out_file + 'revision_text.pkl', 'wb'))

if __name__ == '__main__':
    go(in_file)




