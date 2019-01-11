import nltk
import numpy as np



import nltk
import numpy as np

from nltk.parse.stanford import StanfordDependencyParser

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


LEXICONS = [
    'assertives',
    'entailed_arg',
    'entailed',
    'entailing_arg',
    'entailing',
    'factives',
    'hedges',
    'implicatives',
    'negatives',
    'positives',
    'npov',
    'reports',
    'strong_subjectives',
    'weak_subjectives' 
]


class Featurizer:

    def __init__(self, tok2id, pad_id=0, lexicon_feature_bits=1):
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

        # self.parser = StanfordDependencyParser(
        #     model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")    

        self.lexicons = {
            'assertives': self.read_lexicon('lexicons/assertives_hooper1975.txt'),
            'entailed_arg': self.read_lexicon('lexicons/entailed_arg_berant2012.txt'),
            'entailed': self.read_lexicon('lexicons/entailed_berant2012.txt'), 
            'entailing_arg': self.read_lexicon('lexicons/entailing_arg_berant2012.txt'), 
            'entailing': self.read_lexicon('lexicons/entailing_berant2012.txt'), 
            'factives': self.read_lexicon('lexicons/factives_hooper1975.txt'),
            'hedges': self.read_lexicon('lexicons/hedges_hyland2005.txt'),
            'implicatives': self.read_lexicon('lexicons/implicatives_karttunen1971.txt'),
            'negatives': self.read_lexicon('lexicons/negative_liu2005.txt'),
            'positives': self.read_lexicon('lexicons/positive_liu2005.txt'),
            'npov': self.read_lexicon('lexicons/npov_lexicon.txt'),
            'reports': self.read_lexicon('lexicons/report_verbs.txt'),
            'strong_subjectives': self.read_lexicon('lexicons/strong_subjectives_riloff2003.txt'),
            'weak_subjectives': self.read_lexicon('lexicons/weak_subjectives_riloff2003.txt')
        }
        self.lexicon_feature_bits = lexicon_feature_bits


    def read_lexicon(self, fp):
        out = set([
            l.strip() for l in open(fp, errors='ignore') 
            if not l.startswith('#') and not l.startswith(';')
            and len(l.strip().split()) == 1
        ])
        return out


    def lexicon_features(self, words, bits=2):
        assert bits in [1, 2]
        if bits == 1:
            true = 1
            false = 0
        else:
            true = [1, 0]
            false = [0, 1]
    
        out = []
        for word in words:
            out.append([
                true if word in self.lexicons[lex_name] else false
                for lex_name in LEXICONS
            ])
        out = np.array(out)

        if bits == 2:
            out = out.reshape(len(words), -1)

        return out


    def parse_features(self, words):
        # words_tags_rels = []
        # for tree in self.parser.raw_parse(' '.join(words)):
        #     conll = tree.to_conll(4)
        #     conll = [l.split('\t') for l in conll.strip().split('\n')]
        #     words_tags_rels += [(word, tag, rel) for [word, tag, _, rel] in conll]

        # +1 for missing tags
        out_pos = np.zeros((len(words), len(self.pos2id) + 1))
        out_rels = np.zeros((len(words), len(self.rel2id) + 1))
        return out_pos, out_rels
                
        tagi = 0
        for wi, word in enumerate(words):
            if tagi < len(words_tags_rels):
                tagged_word, pos, rel = words_tags_rels[tagi]
            else:
                tagged_word = ' skip me '

            if tagged_word == word:
                out_pos[wi, self.pos2id.get(pos, self.pos2id['<UNK>'])] = 1
                out_rels[wi, self.rel2id.get(rel, self.rel2id['<UNK>'])] = 1
                tagi += 1
            else:
                out_pos[wi, len(self.pos2id)] = 1
                out_rels[wi, len(self.rel2id)] = 1

        return out_pos, out_rels

    def context_features(self, lex_feats, window_size=2):
        out = []
        nwords = lex_feats.shape[0]
        nfeats = lex_feats.shape[1]
        for wi in range(lex_feats.shape[0]):
            window_start = max(wi - window_size, 0)
            window_end = min(wi + window_size + 1, nwords)

            left = lex_feats[window_start: wi, :] if wi > 0 else np.zeros((1, nfeats))
            right = lex_feats[wi + 1: window_end, :] if wi < nwords - 1 else np.zeros((1, nfeats))

            out.append((np.sum(left + right, axis=0) > 0).astype(int))

        return np.array(out)


    def features(self, id_seq):
        if self.pad_id in id_seq:
            pad_len = len(id_seq[id_seq.index(self.pad_id):])
            id_seq = id_seq[:id_seq.index(self.pad_id)]
        else:
            pad_len = 0

        toks = [self.id2tok[x] for x in id_seq]
        # build list of [word, [tok indices the word came from]]
        words = []
        word_indices = []
        for i, tok in enumerate(toks):
            if tok.startswith('##'):
                words[-1] += tok.replace('##', '')
                word_indices[-1].append(i)
            else:
                words.append(tok)
                word_indices.append([i])

        # get features
        pos_feats, rel_feats = self.parse_features(words)
        lex_feats = self.lexicon_features(words, bits=self.lexicon_feature_bits)
        context_feats = self.context_features(lex_feats)

        feats = np.concatenate((pos_feats, rel_feats, lex_feats, context_feats), axis=1)
        # break word-features into tokens
        feats = np.concatenate([
            np.repeat(np.expand_dims(word_vec, axis=0), len(indices), axis=0) 
            for (word_vec, indices) in zip(feats, word_indices)
        ], axis=0)
        feats = np.concatenate((feats, np.zeros((pad_len, feats.shape[1]))))
        return feats


    def featurize_batch(self, batch_ids, padded_len=0):
        """ takes [batch, len] returns [batch, len, features] """
        batch_feats = [self.features(list(id_seq)) for id_seq in batch_ids]
        batch_feats = np.array(batch_feats)
        return batch_feats


