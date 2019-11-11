"""
use stanford coreNLP to make POS + RELATION tags for a copusfile

!!!!!!!!! NOTE: failed lines will simple output "SKIPPED"

might need to 
export JAVAHOME=/orange/brew/data/bin/java 
python tag_corpusfile.py /home/rpryzant/persuasion/data/v5/word_tight/biased.train /home/rpryzant/persuasion/src/multi_disc_tagg/stanford_parser out
python tag_corpusfile.py ../../data/v4/tok/biased.train.pre ../../src/multi_disc_tagg/stanford_parser/ test_prefix
"""
import sys
import os
import numpy as np
import nltk
from tqdm import tqdm
from nltk.parse.stanford import StanfordDependencyParser


corpus_path = sys.argv[1]
# in that dir:
#   wget http://nlp.stanford.edu/software/stanford-parser-full-2015-12-09.zip
#   unzip stanford-parser-full-2015-12-09.zip
stanford_tools_dir = sys.argv[2]

out_path = sys.argv[3]

os.environ['STANFORDTOOLSDIR'] = stanford_tools_dir
os.environ['CLASSPATH'] = '%s/stanford-parser-full-2015-12-09/stanford-parser.jar:%s/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar' % (
    stanford_tools_dir, stanford_tools_dir)
# os.environ['JAVAHOME'] = '/orange/brew/data/bin/java:' + os.environ['JAVAHOME']


# http://universaldependencies.org/docsv1/en/dep/index.html
UD_RELATIONS = [
    'acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux',
    'auxpass', 'case', 'cc', 'cc:preconj', 'ccomp', 'compound',
    'compound:prt', 'conj', 'cop', 'csubj', 'csubjpass', 'dep',
    'det', 'det:predet', 'discourse', 'dislocated', 'dobj',
    'expl', 'foreign', 'goeswith', 'iobj', 'list', 'mark', 'mwe',
    'name', 'neg', 'nmod', 'nmod:npmod', 'nmod:poss', 'nmod:tmod',
    'nsubj', 'nsubjpass', 'nummod', 'parataxis', 'punct', 'remnant',
    'reparandum', 'root', 'vocative', 'xcomp'
]
rel2id = {x: i for i, x in enumerate(UD_RELATIONS)}
rel2id['<UNK>'] = len(rel2id)

pos2id = {
    key: idx for idx, key in enumerate(
        nltk.data.load('help/tagsets/upenn_tagset.pickle').keys())
}
pos2id['<UNK>'] = len(pos2id)



############################## TIMOUT
from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
        pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
            
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator
############################## TIMOUT



def words_from_toks(toks):
    words = []
    word_indices = []
    for i, tok in enumerate(toks):
        if tok.startswith('##'):
            words[-1] += tok.replace('##', '')
            word_indices[-1].append(i)
        else:
            words.append(tok)
            word_indices.append([i])
    return words, word_indices

@timeout(10)
def pos_rel_from_words(words, word_indices):
    words_tags_rels = []
    # TODO MOVE THIS OUTSIDE!! CLEAN WORDS OF BROKEN ENCODINGS!!
    try:
        trees = parser.raw_parse(' '.join(words))
    except:
        print('SKIPPING...')
        return ['SKIPPED'], ['SKIPPED']

    for tree in trees:
        conll = tree.to_conll(4)
        conll = [l.split('\t') for l in conll.strip().split('\n')]
        words_tags_rels += [(word, tag, rel) for [word, tag, _, rel] in conll]

    # +1 for missing tags
    out_pos = []
    out_rels = []

    tagi = 0
    for (wi, word), indices in zip(enumerate(words), word_indices):
        if tagi < len(words_tags_rels):
            tagged_word, pos, rel = words_tags_rels[tagi]
        else:
            tagged_word = ' skip me '

        if tagged_word == word:
            for _ in range(len(indices)):
                    out_pos.append(pos)
                    out_rels.append(rel)
            tagi += 1
        else:
            for _ in range(len(indices)):
                    out_pos.append('<SKIP>')
                    out_rels.append('<SKIP>')

    return out_pos, out_rels


parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")    

out_path = open(out_path + '.tsv', 'w')

for l in tqdm(open(corpus_path), total=sum(1 for line in open(corpus_path))):
    parts = l.strip().split('\t')
    if len(parts) != 2:
            continue
    [l_pre, l_post] = parts
    toks = l_pre.strip().split()

    words, word_indices = words_from_toks(toks)
    try:
            pos, rel = pos_rel_from_words(words, word_indices)
    except TimeoutError:
            pos, rel = ['SKIPPED'], ['SKIPPED']
    print(len(pos), len(rel), len(toks))

    out_path.write('%s\t%s\t%s\t%s\n' % (
            l_pre.strip(), l_post.strip(),
            ' '.join(pos), ' '.join(rel)))





