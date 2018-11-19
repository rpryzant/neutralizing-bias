"""
generates a TSV parallel corpus from a crawl (the output of gain_wiki_revision.py)

python gen_parallel_corpus.py [root dir with all of the yearly wikipedia files] [out path]
"""

import sys
import os
import pickle
from itertools import groupby
import random
import mwparserfromhell
import re
from nltk import sent_tokenize


METADATA_PATTERN = 'en_npov_edits_%d.tsv'
REVISIONS_PATTERN = 'en_npov_edits_%d.revision_text.wo.pkl'

CTR_ID_MISMATCH = 0
CTR_MULTIPLE_EDITS = 0
CTR_NO_TEXT = 0 
CTR_NO_EDITS = 0
CTR_SENT_MISMATCH = 0

wiki_root = sys.argv[1]
out_path = sys.argv[2]


def load_metadata(path):
    out = {}
    for l in open(path):
        parts = l.strip().split('\t')
        out[parts[0]] = {
            'rev_comment': parts[1],
            'rev_user': parts[2],
            'rev_user_text': parts[3],
            'rev_timestamp': parts[4],
            'rev_minor_edit': parts[5]
        }
    return out


def diff(prev_str, next_str):
    prev_set = set(prev_str.split())
    next_set = set(next_str.split())
    return prev_set.symmetric_difference(next_set)


def get_sents(prev_edit_str, next_edit_str):
    global CTR_SENT_MISMATCH

    prev_sents = sent_tokenize(prev_edit_str)
    next_sents = sent_tokenize(next_edit_str)

    if len(prev_sents) != len(next_sents):
        CTR_SENT_MISMATCH += 1
        return
    
    for i, (prev_sent, next_sent) in enumerate(zip(prev_sents, next_sents)):
        diff_size = len(diff(prev_sent, next_sent))
        if diff_size > 0:
            prev_ctx = prev_sents[i - 1] if i > 0 else ''
            post_ctx = prev_sents[i + 1] if i < len(prev_sents) - 1 else ''
            yield prev_sent, next_sent, prev_ctx + ' || ' + post_ctx


def prep_wikitext(token_list):    
    x = ' '.join(token_list)
    # fix tags
    x = x.replace('< ', '<')
    x = x.replace('</ ', '</')
    x = x.replace(' >', '>')
    x = x.replace(' />', '/>')

    parse = mwparserfromhell.parse(x)
    plaintext = parse.strip_code()
    
    # fix pre-tokenization errors
    # replace links with their name
    m = re.match('\[{2}.*\|(.*)\]{2}', plaintext)
    if m:
        plaintext = re.sub('\[{2}.*\|(.*)\]{2}', m.group(1), plaintext)

    # Othwise get rid of the links (no name)
    plaintext = plaintext.replace('[[', '')
    plaintext = plaintext.replace(']]', '')
    
    # rm [urls] and urls
    plaintext = re.sub('\[.*?\]', '', plaintext)
    # TODO -- tokenized urls 
    # collapse multispaces 
    plaintext = re.sub('[ ]+', ' ', plaintext)

    return plaintext


def extract_examples(metadata, revisions):
    global CTR_ID_MISMATCH
    global CTR_MULTIPLE_EDITS
    global CTR_NO_TEXT
    global CTR_NO_EDITS


    for i, (rev_id, metadata_dict) in enumerate(iter(metadata.items())):
        # ignore headers...     
        if i == 0: continue 
        
        if rev_id not in revisions:
            CTR_ID_MISMATCH += 1
            continue
            
        prevs, nexts = revisions[rev_id]

        if 0 in prevs or 0 in nexts:
            CTR_MULTIPLE_EDITS += 1
            continue

        prev_text = prep_wikitext(prevs)
        next_text = prep_wikitext(nexts)

        if not prev_text or not next_text:
            CTR_NO_TEXT += 1
            continue
        
        i = 0
        for prev_sent, next_sent, context in get_sents(prev_text, next_text):
            i += 1
            yield (
                rev_id, metadata_dict['rev_comment'],
                prev_sent, next_sent, context
            )

        if i == 0: 
            CTR_NO_EDITS += 1


examples = []
for year in range(2008, 2019):
    md_path = os.path.join(wiki_root, METADATA_PATTERN % year)
    rev_path = os.path.join(wiki_root, REVISIONS_PATTERN % year)

    try:
        metadata = load_metadata(md_path)
        revisions = pickle.load(open(rev_path, 'rb'))
    except FileNotFoundError:
        continue

    examples += [ex for ex in extract_examples(metadata, revisions)]
    print('=' * 80)
    print(year)
    print('Examples ', len(examples))
    print('id mismatch ', CTR_ID_MISMATCH)
    print('multiple edits ', CTR_MULTIPLE_EDITS)
    print('no text ', CTR_NO_TEXT)
    print('sent mismatch ', CTR_SENT_MISMATCH)


with open(out_path, 'w') as f:
    for ex in examples:
        f.write('\t'.join(ex) + '\n')







