"""
implements the regression baseline from
    http://www.aclweb.org/anthology/P13-1162

python tagging/baseline.py --train ../../data/v6/corpus.wordbiased.tag.train --test ../../data/v6/corpus.wordbiased.tag.test --working_dir TEST --train_batch_size 128 --test_batch_size 128
"""
import sys
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import scipy


from utils import is_ranking_hit
from features import Featurizer
import sys; sys.path.append('.')
from shared.data import get_dataloader
from shared.args import ARGS




if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)



print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

train_dataloader, num_train_examples = get_dataloader(
    ARGS.train,
    tok2id, ARGS.train_batch_size, ARGS.max_seq_len, ARGS.working_dir + '/train_data.pkl')
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, ARGS.test_batch_size, ARGS.max_seq_len, ARGS.working_dir + '/test_data.pkl',
    test=True)

featurizer = Featurizer(tok2id)


def data_for_scipy(dataloader, by_seq=False):
    outX = []
    outY = []
    for batch in tqdm(dataloader):
        ( 
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            tok_label_id, _, tok_dist,
            replace_id, rel_ids, pos_ids, type_ids, categories
        ) = batch   
        pre_id = pre_id.numpy()
        pre_len = pre_len.numpy()
        rel_ids = rel_ids.numpy()
        pos_ids = pos_ids.numpy()
        tok_label_id = tok_label_id.numpy()

        features = featurizer.featurize_batch(pre_id, rel_ids, pos_ids)
        for id_seq, seq_feats, seq_len, label_seq in zip(pre_id, features, pre_len, tok_label_id):
            seqX = []
            seqY = []
            for ti in range(seq_len):
                word_features = np.zeros(len(tok2id))
                word_features[id_seq[ti]] = 1.0

                timestep_vec = seq_feats[ti]
                #timestep_vec = np.concatenate((word_features, seq_feats[ti]))

                seqX.append(csr_matrix(timestep_vec))
                seqY.append(label_seq[ti])

            if by_seq:
                outX.append(scipy.sparse.vstack(seqX))
                outY.append(seqY)
            else:
                outX += seqX
                outY += seqY
    if by_seq:
        return outX, outY

    return scipy.sparse.vstack(outX), outY

trainX, trainY = data_for_scipy(train_dataloader, by_seq=False)
# trainX, trainY = data_for_scipy(eval_dataloader, by_seq=False)
testX, testY = data_for_scipy(eval_dataloader, by_seq=True)
trainX, trainY = shuffle(trainX, trainY)

print('TRAINIG...')
model = LogisticRegression()
model.fit(trainX, trainY)

print('TESTING...')
hits, total = 0, 0
for seqX, seqY in tqdm(zip(testX, testY)):
    Y_proba = model.predict_proba(seqX)
    hits += is_ranking_hit(Y_proba, seqY)
    total += 1

print('ACC: ', hits * 1.0 / total)





