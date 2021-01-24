"""
Take the output of get_revision_ids.py and download
revisions from wikipedia. Store the outputs as a tsv with
columns

id       prev      next        prev    next
       (modified chunks)    (singleton chunks)

where 
    "modified chunks" = chunks on the diff page where the enclosed text was changed
    "singleton chunks" = chunks on the diff page where the enclosed text only 
                        occurs on the right or left side (inplying that the editor
                        simply deleted or added that chunk of text)

"""


import re
import sys
import csv
import operator
import numpy as np
import string, pickle, os
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
from tqdm import tqdm
import mwparserfromhell
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen
import codecs



in_file = sys.argv[1]
out_file = sys.argv[2]

# special characters
separator = 0
mask_char = 1 
unknown   = 2
to_TBD    = 3
offset    = 4

# colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_withcolor(idx, l):
    l = l.replace('\n', '')
    ins_p = re.compile(r'<ins.*?>(.*?)</ins>', re.DOTALL)
    del_p = re.compile(r'<del.*?>(.*?)</del>', re.DOTALL)
    patterns = [ins_p, del_p]
    for i,p in enumerate(patterns):
        match = re.finditer(p, l)
        if match:
            new_l = ""
            last = 0
            for m in match:
                if i == 1:
                    color = bcolors.OKBLUE
                else:
                    color = bcolors.OKGREEN
                new_l = new_l + l[last:m.start(1)] + color + m.group(1) + bcolors.ENDC
                last = m.end(1)
            new_l = new_l + l[last:]
            l = new_l
    print (bcolors.HEADER+'line '+str(idx+1)+':'+bcolors.ENDC+l)





def html2diff(html):
    prev_changed, next_changed = [],[]
    prev_deleted, next_added = [],[]

    soup = BeautifulSoup(html, 'html')
    nodes = soup.find_all(class_=re.compile(r'(diff-deletedline)|(diff-addedline)|(diff-empty)'))
    div_p = re.compile(r'<div.*?>(.*)</div>', re.DOTALL)
    
    for i in range(0, len(nodes), 2):
        # skip straddeling cases
        if i + 1 >= len(nodes):
            continue
    
        node_prev = nodes[i]
        node_next = nodes[i + 1]

        # seperate  revisions into chunks that were modified,
        # chunks that were purely deleted and chunks that were purely added
        if not node_prev.div and not node_next.div:
            continue
        elif not node_prev.div:
            next_match = re.match(div_p, node_next.div.prettify(formatter=None))
            if next_match:
                a = next_match.group(1).strip()
                b = a.encode('utf8').decode('utf8')
                next_added.append(b)
        elif not node_next.div:
            prev_match = re.match(div_p, node_prev.div.prettify(formatter=None))
            if prev_match:
                a = prev_match.group(1).strip()
                b = a.encode('utf8').decode('utf8')
                prev_deleted.append(b)
        else:
            prev_match = re.match(div_p, node_prev.div.prettify(formatter=None))
            next_match = re.match(div_p, node_next.div.prettify(formatter=None))
            if prev_match and next_match:
                a = prev_match.group(1).strip()
                b = a.encode('utf8').decode('utf8')
                prev_changed.append(b)

                a = next_match.group(1).strip()
                b = a.encode('utf8').decode('utf8')

                file = codecs.open("lol", "w", "utf-8")
                file.write(b)
                file.close()

                next_changed.append(b)

    return prev_changed, next_changed, prev_deleted, next_added


def url2diff(url):
    try:
        response = urlopen(url)
        html = response.read()
        return html2diff(html)
    except Exception as e:
        print(e, file=sys.stderr)
        return [], [], [], []


def wiki_text_clean(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    return text

def gen_revisions(rev_ids):
    rev_size = len(rev_ids)
    success = 0
    out = {}

    for rev_id in tqdm(rev_ids):
        print('processing revision id = ' + str(rev_id), file=sys.stderr)

        url = 'https://fr.wikipedia.org/wiki/?diff=' + str(rev_id) # changed from https://en.wikipedia.org/wiki/?diff=
        prevs_, nexts_, prev_deleted, next_added = url2diff(url)

        if len(prevs_) != len(nexts_):
            print('ERROR: corpus sizes not equal!', file=sys.stderr)
            continue
            
        prevs, nexts = [], []

        for pre, post in zip(prevs_, nexts_):
            prevs.append( wiki_text_clean(pre) )
            nexts.append( wiki_text_clean(post) )
        prevs_deleted = [wiki_text_clean(pre) for pre in (prev_deleted or ['no_deleted_chunks'])]
        nexts_added = [wiki_text_clean(nxt) for nxt in (next_added or ['no_added_chunks'])]

        if len(prevs) > 0 and len(nexts) > 0:
            print('...success!', file=sys.stderr)
            success += 1
            yield rev_id, prevs, nexts, prevs_deleted, nexts_added

    print('failures: ', rev_size - success, file=sys.stderr)

    return out


def go(in_file, out_file):
    print(in_file)
    print(out_file)
    with codecs.open(in_file, 'r') as f:
        rev_ids = [l.split('\t')[0] for l in f]

    f = codecs.open(out_file, "w", "utf-8")
    for rev_id, prevs, nexts, prev_deleted, next_added in gen_revisions(rev_ids):
        prevs_new = ('<EDIT-DELIM>'.join(prevs)).encode('utf8').decode('utf8')
        nexts_new = ('<EDIT-DELIM>'.join(nexts)).encode('utf8').decode('utf8')
        prev_d_new = ('<EDIT-DELIM>'.join(prev_deleted)).encode('utf8').decode('utf8')
        next_a_new = ('<EDIT-DELIM>'.join(next_added)).encode('utf8').decode('utf8')
        output = '\t'.join([rev_id, prevs_new, nexts_new, prev_d_new, next_a_new]) + "\n"
        f.write(output)
    
    f.close()


if __name__ == '__main__':
    go(in_file, out_file)
