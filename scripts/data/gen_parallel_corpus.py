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
from nltk import sent_tokenize, word_tokenize
import diff_match_patch as dmp_module
import Levenshtein
import numpy as np

METADATA_PATTERN = 'en_npov_edits_%d.tsv'
REVISIONS_PATTERN = 'en_npov_edits_%d.revision_text.wo.pkl'

CTR_ID_MISMATCH = 0
CTR_MULTIPLE_EDITS = 0
CTR_NO_TEXT = 0 
CTR_NO_EDITS = 0
CTR_SENT_MISMATCH = 0
CTR_RATIO_SKIPPED = 0

wiki_root = sys.argv[1]
out_path = sys.argv[2]


def diff(s1, s2):
    dmp = dmp_module.diff_match_patch()
    d = dmp.diff_main(s1, s2)
    dmp.diff_cleanupSemantic(d)
    return d


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




def get_sents(prev_edit_str, next_edit_str):
    global CTR_SENT_MISMATCH

    prev_sents = sent_tokenize(prev_edit_str)
    next_sents = sent_tokenize(next_edit_str)

    if len(prev_sents) != len(next_sents):
        CTR_SENT_MISMATCH += 1
        return
    
    for i, (prev_sent, next_sent) in enumerate(zip(prev_sents, next_sents)):
        sent_diff = diff(prev_sent, next_sent)
        num_shared_regions = len([x for x in sent_diff if x[0] == 0])
        num_dif_regions = len([x for x in sent_diff if x[0] != 0])
        lev_dist = Levenshtein.distance(prev_sent, next_sent)

        if len(sent_diff) > 1 and lev_dist > 4 and num_shared_regions > 0 and num_dif_regions > 0:
            prev_ctx = prev_sents[i - 1] if i > 0 else ''
            post_ctx = prev_sents[i + 1] if i < len(prev_sents) - 1 else ''
            context = prev_ctx.strip() + ' ||| ' + post_ctx.strip()

            example = (
                prev_sent.strip(),
                next_sent.strip(),
                context.strip()
            )
            yield example


def prep_wikitext(token_list):    
    x = ' '.join(token_list)
    # fix tags
    x = x.replace('< ', '<')
    x = x.replace('</ ', '</')
    x = x.replace(' >', '>')
    x = x.replace(' />', '/>')

    parse = mwparserfromhell.parse(x)
    plaintext = parse.strip_code()
    
    # rm [[text]] and [text]
    plaintext = re.sub('\[?\[.*?\]\]?', '', plaintext)
    # rm {{text}} and {text}
    plaintext = re.sub('\{?\{.*?\}\}?', '', plaintext)
    # collapse multispaces 
    plaintext = re.sub('[ ]+', ' ', plaintext)
    # remove urls
    plaintext = re.sub('(?P<url>https?://[^\\s]+)', '', plaintext)
    # remove wiki headings (sometimes mwparserfromhell misses these)
    plaintext = re.sub('==(.*)?==', '', plaintext)
    # remove leftover table bits
    plaintext = re.sub('thumb\|(right|left)(\|?)', '', plaintext)
    # empty parens
    plaintext = plaintext.replace('()', '')

    # rm examples starting with ! or | (will be thrown out in downstream filtering)
    plaintext = plaintext.strip()
    if plaintext and plaintext[0] in ['!', '|']:
        plaintext = ''

    return plaintext


def ratio(s1, s2):
    return len(s1) * 1.0 / len(s2)


def tokenize(s):
    tok_list = word_tokenize(s.lower())
    s_tok = ' '.join(tok_list)
    return re.sub('[ ]+', ' ', s_tok)


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
            # print(prev_sent)
            # print(next_sent)
            # print(diff(prev_sent, next_sent))
            # print()
        
            i += 1
            yield (
                rev_id, tokenize(metadata_dict['rev_comment']),
                tokenize(prev_sent), tokenize(next_sent), tokenize(context),
                ratio(tokenize(prev_sent), tokenize(next_sent))
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


# ratio thresholding
ratios = [ex[-1] for ex in examples]
N = len(ratios) * 1.0 
mu = np.mean(ratios)
sd = np.std(ratios)


with open(out_path, 'w') as f:
    for ex in examples:
        ratio = ex[-1]
        if (ratio < mu - 1.90 * sd) or (ratio > mu + 1.90 * sd):
            CTR_RATIO_SKIPPED += 1
            continue
        f.write('\t'.join(ex[:-1]) + '\n')

print('Beyond ratio limit: ', CTR_RATIO_SKIPPED)





