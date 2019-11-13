import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

import os
import numpy as np

import sys; sys.path.append('.')
from shared.data import get_dataloader
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import model as joint_model
import utils as joint_utils




assert ARGS.inference_output, "Need to specify inference_output arg!"


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, ARGS.test_batch_size, ARGS.working_dir + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok)


# # # # # # # # ## # # # ## # # MODEL # # # # # # # # ## # # # ## # #

if ARGS.pointer_generator:
    debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id) # 768 = bert hidden size
else:
    debias_model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)


if ARGS.extra_features_top:
    tagging_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    tagging_model = tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    tagging_model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache')
        
        
if ARGS.tagger_checkpoint:
    print('LOADING TAGGER FROM ' + ARGS.tagger_checkpoint)
    tagging_model.load_state_dict(torch.load(ARGS.tagger_checkpoint))
    print('DONE.')
if ARGS.debias_checkpoint:
    print('LOADING DEBIASER FROM ' + ARGS.debias_checkpoint)
    debias_model.load_state_dict(torch.load(ARGS.debias_checkpoint))
    print('DONE.')


joint_model = joint_model.JointModel(
    debias_model=debias_model, tagging_model=tagging_model)

if CUDA:
    joint_model = joint_model.cuda()

if ARGS.checkpoint is not None and os.path.exists(ARGS.checkpoint):
    print('LOADING FROM ' + ARGS.checkpoint)
    # TODO(rpryzant): is there a way to do this more elegantly? 
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    if CUDA:
        joint_model.load_state_dict(torch.load(ARGS.checkpoint))
        joint_model = joint_model.cuda()
    else:
        joint_model.load_state_dict(torch.load(ARGS.checkpoint, map_location='cpu'))
    print('...DONE')


# # # # # # # # # # # # EVAL # # # # # # # # # # # # # #
joint_model.eval()
hits, preds, golds, srcs = joint_utils.run_eval(
    joint_model, eval_dataloader, tok2id, ARGS.inference_output,
    ARGS.max_seq_len, ARGS.beam_width)

print('eval/bleu', seq2seq_utils.get_bleu(preds, golds), 0)
print('eval/true_hits', np.mean(hits), 0)

with open(ARGS.working_dir + '/stats.txt', 'w') as f:
    f.write('eval/bleu %d' % seq2seq_utils.get_bleu(preds, golds))
    f.write('eval/true_hits %d' % np.mean(hits))


