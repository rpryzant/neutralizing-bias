"""
generates a TSV parallel corpus from a crawl (the output of gain_wiki_revision.py)

python gen_parallel_corpus.py [root dir with all of the yearly wikipedia files] [out path]

python gen_parallel_corpus.py ~/Desktop/Wiki_NPOV/ TEST
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
from collections import Counter
import math
from tqdm import tqdm

METADATA_PATTERN = 'en_npov_edits_%d.tsv'
REVISIONS_PATTERN = 'en_npov_edits_%d.revision_text.wo.pkl'

CTR_ID_MISMATCH = 0
CTR_MULTIPLE_EDITS = 0
CTR_NO_TEXT = 0 
CTR_NO_EDITS = 0
CTR_SENT_MISMATCH = 0
CTR_RATIO_SKIPPED = 0

wiki_root = sys.argv[1]
out_prefix = sys.argv[2]


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

def ratio(s1, s2):
    return len(s1) * 1.0 / len(s2)


def tokenize(s):
    tok_list = word_tokenize(s.lower())
    s_tok = ' '.join(tok_list)
    return re.sub('[ ]+', ' ', s_tok)


def BLEU(hyp, ref):
    # get ngram stats
    stats = []
    stats.append(len(hyp))
    stats.append(len(ref))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hyp[i:i + n]) for i in range(len(hyp) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(ref[i:i + n]) for i in range(len(ref) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hyp) + 1 - n, 0]))

    # get bleu from stats
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    bleu = math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

    return 100 * bleu


def get_sents(prev_edit_str, next_edit_str):
    global CTR_SENT_MISMATCH

    prev_sents = sent_tokenize(prev_edit_str)
    next_sents = sent_tokenize(next_edit_str)

    for i, prev_sent in enumerate(prev_sents):
        bleus = [
            (BLEU(prev_sents[i], next_sents[j]), j)
            for j in range(max(i - 5, 0), min(i + 5, len(next_sents) - 1))
            ]
        # corner case: way more prev's than next's
        if not bleus: 
            continue
        match_bleu, match_idx = max(bleus)
        next_sent = next_sents[match_idx]
        # skip perfect matches
        if match_bleu == 100:
            yield prev_sent.strip(), next_sent.strip(), '0'
            continue

        # skip near-perfect matches
        if Levenshtein.distance(prev_sent, next_sent) < 4:
            continue

        sent_diff = diff(prev_sent, next_sent)
        shared_regions = [x for x in sent_diff if x[0] == 0]
        dif_regions = [x for x in sent_diff if x[0] != 0]

        # skip completely different matches (or, again, completely identical)
        if not shared_regions or not dif_regions:
            continue

        # skip matches that are too different
        shared_len = len(' '.join([s for _, s in shared_regions]))
        prev_ratio = shared_len * 1.0 / len(prev_sent)
        post_ratio = shared_len * 1.0 / len(next_sent)
        ratio = (prev_ratio + post_ratio) / 2.0
        if ratio < 0.5:
            continue

        # skip matches where only punctuation is shared
        if not re.findall('\w', ''.join([s for _, s in shared_regions])):
            continue

        # TODO -- skip matches where diff occurs AFTER length threshold
# ALSO MAX LENGTH THRESHOLD, THROW OUT EXAMPLES WHOSE
# DIF IS AFTER MY MAX LENGTH (MAYBE 60 OR SO)


        yield prev_sent.strip(), next_sent.strip(), '1'


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
    plaintext = re.sub('\|?thumb( )?\|(.*)?(right|left)( )?(\|?)', '', plaintext)
    plaintext = plaintext.replace('thumb|', '')
    # empty parens
    plaintext = plaintext.replace('()', '')
    # ignore timestamp sentences
    if 'retrieved on' in plaintext.lower():
        plaintext = ''
    # fuck stars
    plaintext = plaintext.replace('*', '')

# TODO?
# 200px|
# ( , , , , )
# | year = 2002 |accessdate = may 31 , 2006 |url=
# name= '' hrw '' / >
# ( come-and-hear .com/yebamoth/yebamoth_47.html # partb babylonian talmud yevamot 47b . )
#    JUST RM ALL PARENS??



    # rm examples starting with ! or | (will be thrown out in downstream filtering)
    plaintext = plaintext.strip()
    if plaintext.startswith('?') or plaintext.startswith('|'):
        plaintext = ''

    return plaintext



def extract_examples(metadata, revisions):
    global CTR_ID_MISMATCH
    global CTR_MULTIPLE_EDITS
    global CTR_NO_TEXT
    global CTR_NO_EDITS


    for i, (rev_id, metadata_dict) in tqdm(enumerate(iter(metadata.items())), total=len(metadata)):
        # ignore headers...     
        if i == 0: continue 
        
        if rev_id not in revisions:
            CTR_ID_MISMATCH += 1
            continue
            
        prevs, nexts = revisions[rev_id]

        if len(prevs) > 1 or len(nexts) > 1:
            CTR_MULTIPLE_EDITS += 1
            continue
            
        prev_text = prep_wikitext(prevs)
        next_text = prep_wikitext(nexts)

        if not prev_text or not next_text:
            CTR_NO_TEXT += 1
            continue
        
        i = 0
        for prev_sent, next_sent, bias_status in get_sents(prev_text, next_text):
            i += 1
#            print(prev_sent)
#            print(next_sent)
#            print()
            yield (
                rev_id, tokenize(metadata_dict['rev_comment']),
                tokenize(prev_sent), tokenize(next_sent), bias_status,
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
ratios = [ex[-1] for ex in examples if ex[-2] == '1'] # only do thresholding on bias
N = len(ratios) * 1.0 
mu = np.mean(ratios)
sd = np.std(ratios)


out_unbiased = open(out_prefix + '.unbiased', 'w')
out_biased = open(out_prefix + '.biased', 'w')

with open(out_prefix, 'w') as f:
    for ex in examples:
        ratio = ex[-1]
        bias = ex[-2] == '1'
        if bias and ((ratio < mu - 1.96 * sd) or (ratio > mu + 1.96 * sd)):
            CTR_RATIO_SKIPPED += 1
            continue
        if bias:
            out_biased.write('\t'.join(ex[:-1]) + '\n')
        else:
            out_unbiased.write('\t'.join(ex[:-1]) + '\n')

print('Beyond ratio limit: ', CTR_RATIO_SKIPPED)





