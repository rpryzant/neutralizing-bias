"""
generates a TSV parallel corpus from a crawl (the output of gain_wiki_revision.py)

python gen_data_from_crawl.py wiki_crawl/final_data.pkl CACHE OUT

pickle_path = sys.argv[1]
cache_path = sys.argv[2]
out_prefix = sys.argv[3]


ONLY TAKE REVISIONS WITH 1 CHANGED SENTENCE?

TODO REMAKDE DATA FROM THE BEGINNING, MAKE SURE TAGS ARE GETTING IN THERE SO YOU CAN DELETE THEM,
        THEN FIDDLE WITH THE LENGTH THRESHOLD!!

# IGNORE IF ONLY URL IS CHANGED, E.G. https://en.wikipedia.org/w/index.php?diff=9779941 (before sent tokenization)

# BETTER URL STRIPPING: http://lcweb2.loc.gov/cgi-bin/query/r?frd/cstdy:@field(DOCID+iq0023
    use "http\S+" ??

[ http : / / www .
( ch : 21 , ve ##r : 104
[ http : / / www . alter ##net .
jericho ( arabic , ar ; standard y ##rio ti ##ber ##ian y ##r / y ##r ; meaning " fra ##grant " . strong ' s bible dictionary greek ' ' ' ' ' ' ) is a town in [ [ palestine ] ] , located within the jericho governorate , near the jordan river .
the azerbaijani ##s ( ; , ) are a [ [ turkic peoples | turkic people ] ] 
col ##sp ##an = 2 | congress body ! !

4 , 09 ##3 killed ##alla ##fr ##ica more than 1 , 700 killed in clashes in 2009 , 1 january 2010 ##iri ##n africa accusations traded over rising casualties at mo ##ga ##dis ##hu market , 2 december 2010 - 6 , 310 < br >
4 , 09 ##3 killed ##alla ##fr ##ica more than 1 , 700 killed in clashes in 2009 , 1 january 2010 ##iri ##n africa accusations traded over rising casualties at mo ##ga ##dis ##hu market , 2 december 2010 - 6 , 310 < < ins class = " di ##ff ##chang ##e di ##ff ##chang ##e - inline
    ===> REMOVE ALL <TAGS> and <BR> and <INCOMPLETE TAGS

URL STUFF LIKE these
118482450   | image name=  <del class="diffchange diffchange-inline">   Joe_McCarthy.JPG  </del>  |  <del class="diffchange diffchange-inline">   220px  </del><EDIT-DELIM>''' Joseph Raymond McCarthy''' ([[November 14]], [[1908]]  [[May 2]], [[1957]]) was a [[Republican Party (United States)|Republican]] [[United States Senate|U.S. Senator]] from the state of [[Wisconsin]] between 1947 and 1957. Beginning in 1950, McCarthy became the most visible public face of a period of extreme [[anti-communism|anti-communist]] suspicion inspired by the  <del class="diffchange diffchange-inline">   later proven presence  </del>  of  <del class="diffchange diffchange-inline">   Soviet spies placed in high ranking government positions in  </del>  the  <del class="diffchange diffchange-inline">   United  </del>  <del class="diffchange diffchange-inline">   States  </del>  . He was noted for making  <del class="diffchange diffchange-inline">   substantiated  </del>  claims that there were large numbers of  <del class="diffchange diffchange-inline">   members of the American  </del>  Communist  <del class="diffchange diffchange-inline">   Party  </del>  and [[Soviet Union|Soviet]] spies and sympathizers inside the federal government  <del class="diffchange diffchange-inline">   . These claims were later substantiated by the findings of the top secret Vernona Project conducted by military intelligence. The purpose of the project was to decrypt the U.S.S.R.'s encrypted cables sent to their spies  </del>  . Ultimately, his tactics  <del class="diffchange diffchange-inline">   were so affective that they  </del>  led to  <del class="diffchange diffchange-inline">   the wide disaproval of the liberal media although  </del>  his  <del class="diffchange diffchange-inline">   popularity  </del>  <del class="diffchange diffchange-inline">   rating was still strong  </del>  and  <del class="diffchange diffchange-inline">   he was widely supported  </del>  by  <del class="diffchange diffchange-inline">   J.  </del>  <del class="diffchange diffchange-inline">   Edgar  </del>  <del class="diffchange diffchange-inline">   Hoover and all US citizens who loved their  </del>  <del class="diffchange diffchange-inline">   country  </del>  . The term "[[McCarthyism]]," coined in 1950 in reference to McCarthy's practices, was soon applied to similar anti-communist pursuits. Today the term is used more generally to describe demagogic, reckless, and unsubstantiated accusations, as well as public attacks on the character or patriotism of political opponents.  <del class="diffchange diffchange-inline">   This is what he was advertised as by the communist media and members of the American Communist Party. It it also noteworthy that the Vernona Projects findings have determined that not only were there 57 Soviet agents, nor a mere 205 Soviet agents but over 300 Soviet agents in high level government positions. One of these agents namedly Alger Hiss advised President Franklin D. Roosevelt at the treaty of Yalta where F.D.R notoriosly gave Poland over to the Soviet Union. The full extent of the massive damage that was incurred by just this one agent is still not fully known, much less the damage caused by over 300 Soviet spies. Thanks to Julius and Ethyl Rosenberg Stalin found out about the existence of the atomic bomb before President Truman did.  </del><EDIT-DELIM>Born and raised on a Wisconsin farm, McCarthy earned a law degree at [[Marquette University]] in 1935 and was elected as a [[Circuit (subnational entity)|circuit]] judge in 1939, the youngest in state history.<EDIT-DELIM>** [  <del class="diffchange diffchange-inline">   http://web.archive.org/web/20051102065730/  </del>  http://www.thenewamerican.com/tna/2003/06-16-2003/vo19no12_witches.htm McCarthy's "Witches']<EDIT-DELIM>**[http://www.  <del class="diffchange diffchange-inline">   jbs  </del>  .  <del class="diffchange diffchange-inline">   org  </del>  /  <del class="diffchange diffchange-inline">   node  </del>  /  <del class="diffchange diffchange-inline">   632  </del>  The Real McCarthy Record] | image name=  <ins class="diffchange diffchange-inline">   Joseph  </ins>  <ins class="diffchange diffchange-inline">   McCarthy.jpg  </ins>  |  <ins class="diffchange diffchange-inline">   200px  </ins><EDIT-DELIM>''' Joseph Raymond McCarthy''' ([[November 14]], [[1908]]  [[May 2]], [[1957]]) was a [[Republican Party (United States)|Republican]] [[United States Senate|U.S. Senator]] from the state of [[Wisconsin]] between 1947 and 1957. Beginning in 1950, McCarthy became the most visible public face of a period of extreme [[anti-communism|anti-communist]] suspicion inspired by the  <ins class="diffchange diffchange-inline">   tensions  </ins>  of the  <ins class="diffchange diffchange-inline">   [[Cold  </ins>  <ins class="diffchange diffchange-inline">   War]]  </ins>  . He was noted for making  <ins class="diffchange diffchange-inline">   unsubstantiated  </ins>  claims that there were large numbers of  <ins class="diffchange diffchange-inline">   [[  </ins>  Communist  <ins class="diffchange diffchange-inline">   party|Communists]]  </ins>  and [[Soviet Union|Soviet]] spies and sympathizers inside the federal government. Ultimately, his tactics led to his  <ins class="diffchange diffchange-inline">   being  </ins>  <ins class="diffchange diffchange-inline">   discredited  </ins>  and  <ins class="diffchange diffchange-inline">   censured  </ins>  by  <ins class="diffchange diffchange-inline">   the  </ins>  <ins class="diffchange diffchange-inline">   United  </ins>  <ins class="diffchange diffchange-inline">   States  </ins>  <ins class="diffchange diffchange-inline">   Senate  </ins>  . The term "[[McCarthyism]]," coined in 1950 in reference to McCarthy's practices, was soon applied to similar anti-communist pursuits. Today the term is used more generally to describe demagogic, reckless, and unsubstantiated accusations, as well as public attacks on the character or patriotism of political opponents.  <ins class="diffchange diffchange-inline">   <ref>  </ins><EDIT-DELIM>Born and raised on a Wisconsin farm, McCarthy earned a law degree at [[Marquette University]] in 1935 and was elected as a [[Circuit (subnational entity)|circuit]] judge in 1939, the youngest in state history.  <ins class="diffchange diffchange-inline">   <ref>  </ins><EDIT-DELIM>** [http://www.thenewamerican.com/tna/2003/06-16-2003/vo19no12_witches.htm McCarthy's "Witches']<EDIT-DELIM>**[http://www.  <ins class="diffchange diffchange-inline">   thenewamerican  </ins>  .  <ins class="diffchange diffchange-inline">   com  </ins>  /  <ins class="diffchange diffchange-inline">   tna  </ins>  /  <ins class="diffchange diffchange-inline">   1996/vo12no18/vo12no18_mccarthy.htm  </ins>  The Real McCarthy Record]
<ref>http://ap.google  </ins>  .  <ins class="diffchange diffchange-inline">   com/article/ALeqM5hRq5w8y3umhKx7G6U--ibphSlGxgD916R7200</ref>


"""
import sys
import os
import pickle
from itertools import groupby
import random
import mwparserfromhell
import re
from nltk import sent_tokenize, word_tokenize
import Levenshtein
import numpy as np
from collections import Counter
import math
from tqdm import tqdm

from nltk import sent_tokenize, word_tokenize

from pytorch_pretrained_bert.tokenization import BertTokenizer
from simplediff import diff
from spellchecker import SpellChecker
from autocorrect import spell



pickle_path = sys.argv[1]
cache_path = sys.argv[2]
out_prefix = sys.argv[3]


CTR_EMPTY_REV = 0
CTR_MULTIPLE_EDITS = 0
CTR_FAILED_CLEANING = 0
CTR_LOW_BLEU = 0
CTR_LOW_LEVEN = 0
CTR_TOO_MANY_1_TOKS = 0
CTR_SPELLING = 0
CTR_FALSE_POSITIVE = 0
CTR_LENGTH_RATIO = 0

BERT_MODEL = "bert-base-uncased"
TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=cache_path)







def clean_wikitext(token_list):    
    x = ' '.join(token_list)

    # preemptively delete <ref>'s, etc from source to avoid "p. 34" type stuff from getting in
    x = re.sub('<[\w=" ]+>.*?<\/\w+>', ' ', x)

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

    # ignore lines without text, e.g. ( , , , , )
    if not re.findall('\w', ''.join(plaintext)):
        plaintext = ''

    # rm examples starting with ! or | 
    plaintext = plaintext.strip()
    if plaintext.startswith('?') or plaintext.startswith('|'):
        plaintext = ''

    # remove tabs and newlines (those is our deliminators beeyotch)
    plaintext.replace('\t', ' ')
    plaintext.replace('\n', ' ')
    plaintext.replace('\r', '')

    return plaintext


def find_matches(a_list, b_list, delta=5):
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


def tokenize(s):
    global TOKENIZER
    tok_list = TOKENIZER.tokenize(s.strip())
    return ' '.join(tok_list)



def sent_generator(revisions):
    global CTR_EMPTY_REV
    global CTR_MULTIPLE_EDITS
    global CTR_FAILED_CLEANING

    for rev_id in tqdm(revisions):
        prevs, posts = revisions[rev_id]

        # empty revision
        if not prevs or not posts:
            CTR_EMPTY_REV += 1
            continue
            
        # unicode dat shit
        if isinstance(prevs[0], bytes):
            prevs = [x.decode() for x in prevs]
        if isinstance(posts[0], bytes):
            posts = [x.decode() for x in posts]

        # multiple edits
        if len(prevs) > 1 or len(posts) > 1:
            CTR_MULTIPLE_EDITS += 1
            continue
            
        prev_text = clean_wikitext(prevs).lower()
        post_text = clean_wikitext(posts).lower()

        # failed cleaning
        if not prev_text or not post_text:
            CTR_FAILED_CLEANING += 1
            continue

        prev_sents_raw = sent_tokenize(prev_text)
        post_sents_raw = sent_tokenize(post_text)
        
        prev_sents_tok = [tokenize(s) for s in prev_sents_raw]
        post_sents_tok = [tokenize(s) for s in post_sents_raw]

        for i, j, score in find_matches(prev_sents_tok, post_sents_tok):
            yield prev_sents_raw[i], prev_sents_tok[i], post_sents_raw[j], post_sents_tok[j], score, rev_id

        # no sents


def is_spelling_diff(d):
    """takes a word diff as arg"""
    global SPELLCHECKER

        # only look at the one-word diffs
    if sum([len(chunk) for tag, chunk in d if tag == '-']) > 1:
        return False

    for i, (tag, words) in enumerate(d):
        if tag == '-' and i+1 < len(d) - 1 and len(words) == 1 and d[i+1][0] == '+':
            # is one-word spelling replacement
            correction = spell(words[0])
            if not correction == words[0] and correction in ' '.join(d[i+1][1]):
                return True

    return False


def get_tok_labels(s_diff):
    tok_labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            tok_labels += ['0'] * len(chunk)
        elif tag == '-':
            tok_labels += ['1'] * len(chunk)
        else:
            pass

    return tok_labels


def should_keep(prev_raw, prev_tok, post_raw, post_tok, bleu, rev_id):
    global CTR_LOW_BLEU
    global CTR_LOW_LEVEN
    global CTR_TOO_MANY_1_TOKS
    global CTR_SPELLING

    # KEEP -- exact match
    if bleu == 100:
        return True, None, '0', ['0' for _ in range(len(prev_tok.split()))]
    # clearly not a match
    if bleu < 10.0:
        CTR_LOW_BLEU += 1
        return False, None, None, None
    # too close
    if Levenshtein.distance(prev_tok, post_tok) < 4:
        CTR_LOW_LEVEN += 1
        return False, None, None, None

    tok_diff = diff(prev_tok.split(), post_tok.split())
    tok_labels = get_tok_labels(tok_diff)
    assert len(tok_labels) == len(prev_tok.split())

    # too dissimilar -- less than half of toks shared
    tok_nums = [int(x) for x in tok_labels]
    if ( sum(tok_nums) * 1.0 / len(tok_nums) ) > 0.5:
        CTR_TOO_MANY_1_TOKS += 1
        return False, None, None, None  

    # edit was just fixing a spelling error
    word_diff = diff(word_tokenize(prev_raw), word_tokenize(post_raw))
    if is_spelling_diff(word_diff):
        CTR_SPELLING += 1
        return False, None, None, None

    single_word_edit = sum([len(chunk) for tag, chunk in word_diff if tag == '-']) == 1

    return True, single_word_edit, '1', tok_labels


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

out = []
for pre_post_bleu_id in sent_generator(revisions):
    keep, is_word_edit, sent_label, tok_labels = should_keep(*pre_post_bleu_id)
    # filtered out
    if not keep: continue

    prev_raw, prev_tok, post_raw, post_tok, _, rev_id = pre_post_bleu_id
    length_ratio = len(prev_raw) * 1.0 / len(post_raw)

    # # false edit
    # if is_word_edit is not None and sum([int(x) for x in tok_labels]) == 0:
    #     CTR_FALSE_POSITIVE += 1
    #     continue

    out.append({
        'is_word_edit': is_word_edit,
        'length_ratio': length_ratio,
        'rev_id': rev_id,
        'out_row': '\t'.join([
            rev_id, 
            # should already be done but w/e just to be safe
            prev_tok.strip().replace('\n', ' ').replace('\t', ' '), 
            post_tok.strip().replace('\n', ' ').replace('\t', ' '), 
            prev_raw.strip().replace('\n', ' ').replace('\t', ' '), 
            post_raw.strip().replace('\n', ' ').replace('\t', ' '), 
            sent_label, ' '.join(tok_labels)
        ])
    })

# ratio thresholding
ratios = [x['length_ratio'] for x in out if out['is_word_edit'] is not None]
N = len(ratios) * 1.0 
mu = np.mean(ratios)
sd = np.std(ratios)


print('WRITING...')
# write unbiased
f_unbiased = open(out_prefix + '.unbiased', 'w')
f_biased = open(out_prefix + '.biased', 'w')
f_word = open(out_prefix + '.wordbiased', 'w')
f_length_skipped = open(out_prefix + '.biased_ratioskipped', 'w')

for ex in out:
    if ex['is_word_edit'] is None:
        f_unbiased.write(ex['out_row'] + '\n')
        continue

    # ratio skip
    r = ex['length_ratio']
    if (r < mu - 2.0 * sd) or (r > mu + 2.0 * sd):
        f_length_skipped.write(ex['out_row'] + '\n')
        CTR_LENGTH_RATIO += 1
        continue

    if ex['is_word_edit']:
        f_word.write(ex['out_row'] + '\n')

    f_biased.write(ex['out_row'] + '\n')


            
f_unbiased.close()
f_biased.close()
f_word.close()
f_length_skipped.close()

print('ctrs:')

print('CTR_EMPTY_REV', CTR_EMPTY_REV)
print('CTR_MULTIPLE_EDITS', CTR_MULTIPLE_EDITS)
print('CTR_FAILED_CLEANING', CTR_FAILED_CLEANING)
print('CTR_LOW_BLEU', CTR_LOW_BLEU)
print('CTR_LOW_LEVEN', CTR_LOW_LEVEN)
print('CTR_TOO_MANY_1_TOKS', CTR_TOO_MANY_1_TOKS)
print('CTR_SPELLING', CTR_SPELLING)
print('CTR_FALSE_POSITIVE', CTR_FALSE_POSITIVE)
print('CTR_LENGTH_RATIO', CTR_LENGTH_RATIO)



