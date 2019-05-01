import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

import os
import numpy as np

import sys; sys.path.append('.')
from shared.data import get_dataloader
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.model as seq2seq_model
import joint.model as joint_model

import model as tagging_model
import utils as tagging_utils

assert ARGS.inference_output, 'Need to specify inference_output arg!'


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


model = joint_model.tagging_model


# # # # # # # # # # # # EVAL # # # # # # # # # # # # # #
model.eval()
outputs = tagging_utils.run_inference(
    model, eval_dataloader, tagging_utils.build_loss_fn(), tokenizer)

# print(outputs)
# print(len(outputs['input_toks']))
# print(len(outputs['tok_probs']))
# print(outputs['input_toks'][0])
# print(outputs['tok_probs'][0])
# print(len(outputs['input_toks'][0]))
# print(len(outputs['tok_probs'][0]))

with open(ARGS.inference_output, 'w') as file:
    for i in range(len(outputs['input_toks'])):
        file.write('Sentence: %s\n' % ' '.join(outputs['input_toks'][i]))
        file.write('Token probs: %s\n' % str(outputs['tok_probs'][i]))
        max_pos = np.argmax(outputs['tok_probs'][i])
        prediction = outputs['input_toks'][i][max_pos]

        # Obtain the full word from word piece.
        pos = max_pos + 1
        while outputs['input_toks'][i][pos].startswith('##'):
            prediction = prediction + outputs['input_toks'][i][pos]
            pos += 1
        pos = max_pos - 1
        while prediction.startswith('##'):
            prediction = outputs['input_toks'][i][pos] + prediction
            pos -= 1
        prediction = prediction.replace('##', '')
        file.write('Prediction: %s\n' % prediction)
        file.write('######################################################\n')
