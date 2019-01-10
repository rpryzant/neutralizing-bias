"""
use stanford coreNLP to make POS + RELATION tags for a copusfile

python tag_corpusfile.py ../../data/v4/tok/biased.train.pre ../../src/multi_disc_tagg/stanford_parser/ test_prefix

might need to 
export JAVAHOME=/orange/brew/data/bin/java 
python tag_corpusfile.py /home/rpryzant/persuasion/data/v5/word_tight/biased.train.pre /home/rpryzant/persuasion/src/multi_disc_tagg/stanford_parser test_prefix
"""
import sys
import os
import numpy as np
import nltk
from tqdm import tqdm
from nltk.parse.stanford import StanfordDependencyParser


corpusfile = sys.argv[1]
# in that dir:
#   wget http://nlp.stanford.edu/software/stanford-parser-full-2015-12-09.zip
#   unzip stanford-parser-full-2015-12-09.zip
stanford_tools_dir = sys.argv[2]

out_prefix = sys.argv[3]

os.environ['STANFORDTOOLSDIR'] = stanford_tools_dir
os.environ['CLASSPATH'] = '%s/stanford-parser-full-2015-12-09/stanford-parser.jar:%s/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar' % (
    stanford_tools_dir, stanford_tools_dir)
os.environ['JAVAHOME'] = '/orange/brew/data/bin/java:' + os.environ['JAVAHOME']


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

def pos_rel_from_words(words):
    words_tags_rels = []
    for tree in parser.raw_parse(' '.join(words)):
        conll = tree.to_conll(4)
        conll = [l.split('\t') for l in conll.strip().split('\n')]
        words_tags_rels += [(word, tag, rel) for [word, tag, _, rel] in conll]

    # +1 for missing tags
    out_pos = []
    out_rels = []

    tagi = 0
    for wi, word in enumerate(words):
        if tagi < len(words_tags_rels):
            tagged_word, pos, rel = words_tags_rels[tagi]
        else:
            tagged_word = ' skip me '

        if tagged_word == word:
            out_pos.append(pos)
            out_rels.append(rel)
            tagi += 1
        else:
            out_pos.append('<SKIP>')
            out_rels.append('<SKIP>')

    return out_pos, out_rels


parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")    

out_pos = open(out_prefix + '.pos', 'w')
out_rel = open(out_prefix + '.rel', 'w')

for l in tqdm(open(corpusfile), total=sum(1 for line in open(corpusfile))):
    toks = l.strip().split()

    words, word_indices = words_from_toks(toks)
    pos, rel = pos_rel_from_words(words)
    assert len(words) == len(pos) == len(rel)
    
    out_pos.write(' '.join(pos) + '\n')
    out_rel.write(' '.join(rel) + '\n')





