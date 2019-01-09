import nltk
import numpy as np



import nltk
import numpy as np

from nltk.parse.stanford import StanfordDependencyParser


# http://universaldependencies.org/u/dep/
UD_RELATIONS = [
    'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc',
    'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det',
    'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith',
    'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl',
    'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp'
]




class Featurizer:

    def __init__(self, tok2id, pad_id=0):
        self.tok2id = tok2id
        self.id2tok = {x: tok for tok, x in tok2id.items()}
        self.pad_id = pad_id

        self.pos2id = {
            key: idx for idx, key in enumerate(
                nltk.data.load('help/tagsets/upenn_tagset.pickle').keys())
        }
        self.pos2id['<UNK>'] = len(self.pos2id)
        
        self.rel2id = {x: i for i, x in enumerate(UD_RELATIONS)}
        self.rel2id['<UNK>'] = len(self.rel2id)

        self.parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")    

    def pos_features(self, words):
        tags = [tag for word, tag in nltk.pos_tag(words)]
        tag_ids = [self.tag2id.get(x, len(self.tag2id)) for x in tags]
        out = np.zeros((len(words), len(self.tag2id) + 1))
        for idx, tag_id in enumerate(tag_ids):
            out[idx, tag_id] = 1
        return out

    def parse_features(self, words):
        words_tags_rels = []
        for tree in self.parser.raw_parse(' '.join(words)):
            conll = tree.to_conll(4)
            conll = [l.split('\t') for l in conll.strip().split('\n')]
            words_tags_rels += [(word, tag, rel) for [word, tag, _, rel] in conll]

        # +1 for missing
        out_pos = np.zeros((len(words), len(self.pos2id) + 1))
        out_rels = np.zeros((len(words), len(self.rel2id) + 1))
        
        tagi = 0
        for wi, word in enumerate(words):
            if tagi < len(words_tags_rels):
                tagged_word, pos, rel = words_tags_rels[tagi]
            else:
                tagged_word = ' skip me '

            if tagged_word == word:
                out_pos[wi, self.pos2id.get(pos, self.pos2id['<UNK>'])] = 1
                out_rels[wi, self.rel2id.get(pos, self.rel2id['<UNK>'])] = 1
                tagi += 1
            else:
                out_pos[wi, len(self.pos2id)] = 1
                out_rels[wi, len(self.rel2id)] = 1

        return out_pos, out_rels


    def features(self, id_seq):
        if self.pad_id in id_seq:
            id_seq = id_seq[:id_seq.index(self.pad_id)]
        toks = [self.id2tok[x] for x in id_seq]
        words = []
        for i, tok in enumerate(toks):
            if tok.startswith('##'):
                words[-1] += tok.replace('##', '')
            else:
                words.append(tok)

        pos_feats, rel_feats = self.parse_features(words)
        print(pos_feats.shape)
        print(rel_feats.shape)
        # need
        # POS
        # assertives
        # entailments
        # factives
        # hedges
        # pos/neg
        # implicatives
        # reports
        # npov lexicon
        # subjectives
        # parse


    def featurize_batch(self, batch_ids):
        batch_feats = [self.features(list(id_seq)) for id_seq in batch_ids]
        quit()



