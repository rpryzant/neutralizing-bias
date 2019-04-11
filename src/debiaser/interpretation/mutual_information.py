"""
get mutual information between features and bias

git add interpretation/mutual_information.py --train ../../data/v6/corpus.wordbiased.tag.train --working_dir TEST/ --checkpoint interpretation/models/inference_model_toksm_cov_4.ckpt
"""
import sys; sys.path.append('.')
from shared.args import ARGS
from shared.data import get_dataloader
from tagging.features import Featurizer
import feature_importance


from pytorch_pretrained_bert.tokenization import BertTokenizer

from tqdm import tqdm
from collections import defaultdict
import math

BERT_MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


featurizer = Featurizer(tok2id)
feature_names = featurizer.get_feature_names()

TARGET_VARIABLE = 'entailing'
tgt_idx = feature_names.index(TARGET_VARIABLE)

dataloader, num_pretrain_examples = get_dataloader(
    ARGS.train,
    tok2id,
    ARGS.train_batch_size,
    ARGS.working_dir + '/pretrain_data.pkl')



feature_counts = defaultdict(lambda: {
    'n00': 1.,  # docs without term, 0 label
    'n01': 1.,  # docs without term, 1 label
    'n10': 1.,  # docs with term, 0 label
    'n11': 1.   # docs with term, 1 label
})

for batch in tqdm(dataloader):
    (
        pre_id, pre_mask, pre_len, 
        post_in_id, post_out_id, 
        pre_tok_label_id, post_tok_label_id,
        rel_ids, pos_ids, categories
    ) = batch

    features = featurizer.featurize_batch(
        pre_id.detach().cpu().numpy(), 
        rel_ids.detach().cpu().numpy(), 
        pos_ids.detach().cpu().numpy(), 
        padded_len=pre_id.shape[1])

    for feat_seq, label_seq in zip(features, pre_tok_label_id.detach().cpu().numpy()):
        for feat_vec, label in zip(feat_seq, label_seq):
            # end of sequence -- ignore pads
            if label == 2:
                break

            for feat_val, feat_name in zip(feat_vec, feature_names):
                feature_counts[feat_name]['n%d%d' % (feat_val, label)] += 1

def mi(n00, n01, n10, n11):
    n0_ = n00 + n01   # docs without term
    n1_ = n11 + n10   # docs with term
    n_0 = n10 + n00   # docs with 0 label
    n_1 = n11 + n01   # docs with 1 label
    n = n00 + n01 + n11 + n10   # total n    

    mutual_info = (n11/n) * math.log((n * n11) / (n1_ * n_1)) + \
                  (n01/n) * math.log((n * n01) / (n0_ * n_1)) + \
                  (n10/n) * math.log((n * n10) / (n1_ * n_0)) + \
                  (n00/n) * math.log((n * n00) / (n0_ * n_0))        
    return mutual_info


MIs = dict(map(lambda x : (x[0], mi(**x[1])), feature_counts.items()))
Is = feature_importance.importance_scores(ARGS.checkpoint)

from scipy.stats.stats import pearsonr

print(pearsonr(*zip(*[(MIs[x], Is[x]) for x in MIs if Is[x] > 0])))




            
            
            