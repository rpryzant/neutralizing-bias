"""
generates a TSV parallel corpus from a crawl (the output of gain_wiki_revision.py)

python gen_data_from_crawl.py wiki_crawl/final_data.pkl CACHE OUT

pickle_path = sys.argv[1]
cache_path = sys.argv[2]
out_prefix = sys.argv[3]

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


from pytorch_pretrained_bert.tokenization import BertTokenizer
from simplediff import diff
from spellchecker import SpellChecker



pickle_path = sys.argv[1]
cache_path = sys.argv[2]
out_prefix = sys.argv[3]


BERT_MODEL = "bert-base-uncased"
TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=cache_path)


CTR_MULTIPLE_EDITS = 0
CTR_FAILED_CLEANING = 0 
CTR_SENT_EXTRACTION_FAILED = 0
CTR_RATIO_SKIPPED = 0
CTR_EMPTY_REV = 0
CTR_FILTERED_OUT = 0
CTR_TO_MANY_1_TOK_LABELS = 0
CTR_SPELLING_FIX = 0

from nltk import sent_tokenize, word_tokenize


def diff_dmp(s1, s2):
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


def clean_wikitext(token_list):    
    x = ' '.join(token_list)
    # fix tags
    x = x.replace('< ', '<')
    x = x.replace('</ ', '</')
    x = x.replace(' >', '>')
    x = x.replace(' />', '/>')

    parse = mwparserfromhell.parse(x)
    plaintext = parse.strip_code()
    
    # remove tabs and newlines (those is our deliminators beeyotch)
    plaintext.replace('\t', ' ')
    plaintext.replace('\n', ' ')

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

    # ignore lines without text, e.g. ( , , , , )
    if not re.findall('\w', ''.join(plaintext)):
        plaintext = ''

    # rm examples starting with ! or | 
    plaintext = plaintext.strip()
    if plaintext.startswith('?') or plaintext.startswith('|'):
        plaintext = ''

    return plaintext



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


def find_matches(a_list, b_list, delta=5):
    for i in range(len(a_list)):
        neighborhood_bleus = [
            (BLEU(a_list[i].split(), b_list[j].split()), j)
            for j in range(max(i - delta, 0), min(i + delta, len(b_list) - 1))
        ]
        # corner case: len(a_list) >> len(b_list)
        if not neighborhood_bleus:
            continue
        
        max_bleu, match_idx = max(neighborhood_bleus)
        
        yield i, match_idx, max_bleu

       
     
def is_spelling_diff(prev_sent, post_sent):
    d = diff(word_tokenize(prev_sent), word_tokenize(post_sent))

    # only look at the one-word diffs
    if sum([len(chunk) for tag, chunk in d if tag == '-']) > 1:
        return False

    sp = SpellChecker()
    for i, (tag, words) in enumerate(d):
        # is one-word spelling replacement
        if tag == '-' and \
            i+1 < len(d) - 1 and \
            len(words) == 1 and \
            d[i+1][0] == '+' and \
            not sp.correction(words[0]) == words[0] and \
            sp.correction(words[0]) in ' '.join(d[i+1][1]):

            return True

    return False

    

def should_filter(prev_tok, post_tok, prev_raw, post_raw, tok_labels):
    global CTR_TO_MANY_1_TOK_LABELS
    global CTR_SPELLING_FIX
    
    """ whether we should filter out a sentence pair """
    # skip near-perfect matches
    if Levenshtein.distance(prev_tok, post_tok) < 4:
        return True

    sent_diff = diff_dmp(prev_tok, post_tok)
    shared_regions = [x for x in sent_diff if x[0] == 0]
    dif_regions = [x for x in sent_diff if x[0] != 0]

    # skip completely different matches (or, again, completely identical)
    if not shared_regions or not dif_regions:
        return True

    # skip matches that are too different (more than half the sentence should be shared)
    # more than half of toks have to be shared
    assert len(tok_labels) == len(prev_tok.split())
    tok_nums = [int(x) for x in tok_labels]
    if ( sum(tok_nums) * 1.0 / len(tok_nums) ) > 0.5:
        CTR_TO_MANY_1_TOK_LABELS += 1
        return True

    # skip matches where only punctuation is shared
    if not re.findall('\w', ''.join([s for _, s in shared_regions])):
        return True

    # skip matches that only fixed spelling
    if is_spelling_diff(prev_raw, post_raw):
        CTR_SPELLING_FIX += 1
        return True

    # TODO -- skip matches where diff occurs AFTER length threshold???
    # ALSO MAX LENGTH THRESHOLD, THROW OUT EXAMPLES WHOSE
    # DIF IS AFTER MY MAX LENGTH (MAYBE 60 OR SO)

    return False



def get_tok_labels(s1_toks, s2_toks):
    s_diff = diff(s1_toks, s2_toks)
    tok_labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            tok_labels += ['0'] * len(chunk)
        elif tag == '-':
            tok_labels += ['1'] * len(chunk)
        else:
            pass
    assert len(tok_labels) == len(s1_toks)

    return tok_labels

def tokenize(s):
    global TOKENIZER
    tok_list = TOKENIZER.tokenize(s.strip())
    return ' '.join(tok_list)

def extract_sents(prev_edit_str, post_edit_str, rev_id):
    global CTR_FILTERED_OUT

    # break up into sentences
    prev_sents = sent_tokenize(prev_edit_str)
    post_sents = sent_tokenize(post_edit_str)
    # break sentences into lower-cased tokens
    prevs = [tokenize(s.lower()) for s in prev_sents]
    posts = [tokenize(s.lower()) for s in post_sents]

    for i, j, score in find_matches(prevs, posts):
        cur_prev = prevs[i]
        cur_post = posts[j]

        cur_prev_raw = prev_sents[i]
        cur_post_raw = post_sents[j]

        # perfect match = unchanged (no bias)
        if score == 100:
            if cur_prev == cur_post:
                yield (
                    rev_id, cur_prev, cur_post, cur_prev_raw, cur_post_raw,
                    '0', ' '.join(['0' for _ in range(len(cur_prev.split()))])
                )

        tok_labels = get_tok_labels(cur_prev.strip().split(), cur_post.strip().split())

        if should_filter(cur_prev, cur_post, cur_prev_raw, cur_post_raw, tok_labels):
            CTR_FILTERED_OUT += 1
            continue
        
        yield (
            rev_id, cur_prev, cur_post, cur_prev_raw, cur_post_raw,
            '1', ' '.join(tok_labels)
        )


def extract_examples(revisions):
    global CTR_MULTIPLE_EDITS
    global CTR_FAILED_CLEANING
    global CTR_SENT_EXTRACTION_FAILED
    global CTR_EMPTY_REV

    for rev_id in tqdm(revisions):
        prevs, nexts = revisions[rev_id]
        
        if not prevs or not nexts:
            CTR_EMPTY_REV += 1
            continue
            
        # unicode dat shit
        if isinstance(prevs[0], bytes):
            prevs = [x.decode() for x in prevs]
        if isinstance(nexts[0], bytes):
            nexts = [x.decode() for x in nexts]

        if len(prevs) > 1 or len(nexts) > 1:
            CTR_MULTIPLE_EDITS += 1
            continue
            
        prev_text = clean_wikitext(prevs)
        next_text = clean_wikitext(nexts)

        if not prev_text or not next_text:
            CTR_FAILED_CLEANING += 1
            continue

        i = 0
        for example in extract_sents(prev_text, next_text, rev_id):
            i += 1
            yield example

        if i == 0: 
            CTR_SENT_EXTRACTION_FAILED += 1


def is_single_word_diff(s1, s2):
    s1 = word_tokenize(s1.lower().strip())
    s2 = word_tokenize(s2.lower().strip())
    
    word_diff = diff(s1, s2)
    return sum([len(chunk) for tag, chunk in word_diff if tag == '-']) == 1


# load big pickle 
# https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
print('LOADING PICKLE...')

# revisions = pickle.load(open(pickle_path, 'rb'))
bytes_in = bytearray(0)
max_bytes = 2**31 - 1
input_size = os.path.getsize(pickle_path)
with open(pickle_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
revisions = pickle.loads(bytes_in)

print('EXTRACTING EXAMPLES...')
out_unbiased = []
out_biased = []
#  out_biased_singleword = []    # TODO???
out_biased_singletoken = []

out_all = open(out_prefix + '.all', 'w')

for i, example in enumerate(extract_examples(revisions)):
    # rev_id, prev_toks, post_toks, prev_raw, post_raw, sent_label, tok_labels
    assert len(example) == 7

    # no idea how these characters get in but they're in there...so delete them
    example_row = '\t'.join([x.replace('\t', ' ') for x  in example]).strip().replace('\n', ' ').replace('\r', '')

    out_all.write(example_row + '\n')

    if example[-2] == '0':
        out_unbiased.append(example_row)
    else:
        length_ratio = len(example[3]) * 1.0 / len(example[4])

        out_biased.append( (length_ratio, example_row) )

out_all.close()

# ratio thresholding
ratios = [r for r, _ in out_biased]
N = len(ratios) * 1.0 
mu = np.mean(ratios)
sd = np.std(ratios)


print('WRITING...')
# write unbiased
with open(out_prefix + '.unbiased', 'w') as f:
    for ex in out_unbiased:
        f.write(ex + '\n')

# write biased
f = open(out_prefix + '.biased', 'w')
f_tok = open(out_prefix + '.tokbiased', 'w')
f_word = open(out_prefix + '.wordbiased', 'w')
for r, ex in tqdm(out_biased):
    if (r < mu - 1.96 * sd) or (r > mu + 1.96 * sd):
        CTR_RATIO_SKIPPED += 1
        continue

    f.write(ex + '\n')
    
    ex_parts = ex.strip().split('\t')
    # single tok
    if sum([int(x) for x in ex_parts[-1].split()]) == 1:
        f_tok.write(ex + '\n')

    # single word
    if is_single_word_diff(ex_parts[3], ex_parts[4]):
        f_word.write(ex + '\n')
        
            
f.close()
f_tok.close()
f_word.close()

print('counters:')
print('CTR_MULTIPLE_EDITS', CTR_MULTIPLE_EDITS)
print('CTR_FAILED_CLEANING', CTR_FAILED_CLEANING)
print('CTR_SENT_EXTRACTION_FAILED', CTR_SENT_EXTRACTION_FAILED)
print('CTR_RATIO_SKIPPED', CTR_RATIO_SKIPPED)
print('CTR_EMPTY_REV', CTR_EMPTY_REV)
print('CTR_FILTERED_OUT', CTR_FILTERED_OUT)
print('CTR_TO_MANY_1_TOK_LABELS', CTR_TO_MANY_1_TOK_LABELS)
print('CTR_SPELLING_FIX', CTR_SPELLING_FIX)



