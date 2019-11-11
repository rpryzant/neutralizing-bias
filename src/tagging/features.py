import nltk
import numpy as np



import nltk
import numpy as np

from nltk.parse.stanford import StanfordDependencyParser

import sys; sys.path.append('.')
from shared.data import REL2ID, POS2ID

class Featurizer:

    def __init__(self, tok2id={}, pad_id=0, lexicon_feature_bits=1):
        self.tok2id = tok2id
        self.id2tok = {x: tok for tok, x in tok2id.items()}
        self.pad_id = pad_id

        self.pos2id = POS2ID
        self.rel2id = REL2ID

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


    def get_feature_names(self):

        lexicon_feature_names = list(self.lexicons.keys())
        context_feature_names = [x + '_context' for x in lexicon_feature_names]
        pos_names = list(list(zip(*sorted(self.pos2id.items(), key=lambda x: x[1])))[0])
        rel_names = list(list(zip(*sorted(self.rel2id.items(), key=lambda x: x[1])))[0])

        return lexicon_feature_names + context_feature_names + pos_names + rel_names        

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
                true if word in lexicon else false 
                for _, lexicon in self.lexicons.items()
            ])
        out = np.array(out)

        if bits == 2:
            out = out.reshape(len(words), -1)

        return out


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


    def features(self, id_seq, rel_ids, pos_ids):
        if self.pad_id in id_seq:
            pad_idx = id_seq.index(self.pad_id)
            pad_len = len(id_seq[pad_idx:])
            id_seq = id_seq[:pad_idx]
            rel_ids = rel_ids[:pad_idx]
            pos_ids = pos_ids[:pad_idx]
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

        # get expert features
        lex_feats = self.lexicon_features(words, bits=self.lexicon_feature_bits)
        context_feats = self.context_features(lex_feats)
        expert_feats = np.concatenate((lex_feats, context_feats), axis=1)
        # break word-features into tokens
        feats = np.concatenate([
            np.repeat(np.expand_dims(word_vec, axis=0), len(indices), axis=0) 
            for (word_vec, indices) in zip(expert_feats, word_indices)
        ], axis=0)

        # add in the pos and relational features
        pos_feats = np.zeros((len(pos_ids), len(POS2ID)))
        pos_feats[range(len(pos_ids)), pos_ids] = 1
        rel_feats = np.zeros((len(rel_ids), len(REL2ID)))
        rel_feats[range(len(rel_ids)), rel_ids] = 1
        
        feats = np.concatenate((feats, pos_feats, rel_feats), axis=1)

        # add pad back in                
        feats = np.concatenate((feats, np.zeros((pad_len, feats.shape[1]))))

        return feats


    def featurize_batch(self, batch_ids, rel_ids, pos_ids, padded_len=0):
        """ takes [batch, len] returns [batch, len, features] """

        batch_feats = [
            self.features(list(id_seq), list(rel_ids), list(pos_ids)) 
            for id_seq, rel_ids, pos_ids in zip(batch_ids, rel_ids, pos_ids)]
        batch_feats = np.array(batch_feats)
        return batch_feats


