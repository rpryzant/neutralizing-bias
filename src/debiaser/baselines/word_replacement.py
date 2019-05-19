"""Run word replacement baseline.
1. Get biased word from tagger
2. Pick a replacement word from vocab
3. Compare to ground truth (words where post_tok_labels is 1)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tensorboardX import SummaryWriter
from tqdm import tqdm

import sys; sys.path.append('.')
from shared.args import ARGS
from shared.constants import CUDA
from shared.data import get_dataloader

import joint.model as joint_model
import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils
import tagging.model as tagging_model


class BertForWordReplacement(nn.Module):
    def __init__(self, config, joint_model, tok2id):
        super(BertForWordReplacement, self).__init__()
        self.bert = BertModel.from_pretrained(config)

        self.joint_model = joint_model

        components = [
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, len(tok2id))
        ]
        self.tok_classifier = nn.Sequential(*components)
        
    def forward(self, pre_id, pre_mask, rel_ids=None, pos_ids=None,
                categories=None):

        is_bias_probs, _ = self.joint_model.run_tagger(
            pre_id, pre_mask, rel_ids=rel_ids, pos_ids=pos_ids,
            categories=categories)

        _, bias_pos = is_bias_probs.max(1)
        bias_pos = bias_pos.unsqueeze(1)
        bias_ids = torch.gather(pre_id, 1, bias_pos)

        # TODO: handle multiple tokens when a part of a word is selected
        word_embeddings = self.bert.embeddings(bias_ids).squeeze()

        tok_logits = self.tok_classifier(word_embeddings)
        new_toks = tok_logits.argmax(1).unsqueeze(1)
        new_ids = pre_id.scatter(1, bias_pos, new_toks)

        return tok_logits, new_ids

tokenizer = BertTokenizer.from_pretrained(
    ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)
id2tok = {x: tok for (tok, x) in tok2id.items()}
print("The len of our vocabulary is {}".format(len(tok2id)))

train_dataloader, num_train_examples = get_dataloader(
    ARGS.train,
    tok2id, ARGS.train_batch_size, ARGS.working_dir + '/train_data.pkl',
    categories_path=ARGS.categories_file,
    add_del_tok=ARGS.add_del_tok)
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, ARGS.test_batch_size, ARGS.working_dir + '/test_data.pkl',
    categories_path=ARGS.categories_file,
    test=True, add_del_tok=ARGS.add_del_tok)

if ARGS.pointer_generator:
    debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id) # 768 = bert hidden size
else:
    debias_model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)


tagging_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache',
        tok2id=tok2id)

joint_model = joint_model.JointModel(
    debias_model=debias_model, tagging_model=tagging_model)

if CUDA:
    joint_model = joint_model.cuda()

print('LOADING FROM ' + ARGS.checkpoint)
# TODO(rpryzant): is there a way to do this more elegantly? 
# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
if CUDA:
    joint_model.load_state_dict(torch.load(ARGS.checkpoint))
    joint_model = joint_model.cuda()
else:
    joint_model.load_state_dict(torch.load(ARGS.checkpoint, map_location='cpu'))
print('...DONE')

model = BertForWordReplacement(
    ARGS.bert_model, joint_model, tok2id)

if CUDA:
    model = model.cuda()

params = list(model.tok_classifier.parameters())
params = list(filter(lambda p: p.requires_grad, params))
optimizer = optim.Adam(params, lr=ARGS.learning_rate)
writer = SummaryWriter(ARGS.working_dir)

losses = []
for epoch in range(ARGS.epochs):
    if ARGS.debug_skip and epoch > 2:
        continue
    for step, batch in enumerate(tqdm(train_dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        ( 
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch
        
        tok_logits, _ = model(pre_id, pre_mask,
            rel_ids=rel_ids, pos_ids=pos_ids, categories=categories)

        labels = torch.zeros((ARGS.train_batch_size, len(tok2id)))
        for i in range(len(tok_label_id)):
            for j in range(len(tok_label_id[i])):
                if tok_label_id[i][j].item() == 1:
                    labels[i][post_out_id[i][j].item()] = 1
                elif tok_label_id[i][j].item() == 2:
                    break

            # If there were no changes, then there was a deletion.
            # TODO: is this always true?
            if labels[i].sum() == 0:
                labels[i][-1] = 1

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(tok_logits, labels.cuda())
        loss.backward()
        optimizer.step()
        model.zero_grad()

        losses.append(loss.detach().cpu().numpy())

    print(losses)

    writer.add_scalar('train/loss', np.mean(losses), epoch + 1)

    print('EVAL...')
    model.eval()
    hits, preds, golds, srcs = [], [], [], []
    out_file = None
    out_file = open(ARGS.working_dir + '/results_%d.txt' % epoch, 'w')
    for step, batch in enumerate(tqdm(eval_dataloader)):
        if ARGS.debug_skip and step > 2:
            continue
    
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch

        with torch.no_grad():
            _, predicted_toks = model(
                pre_id, pre_mask, rel_ids=rel_ids, pos_ids=pos_ids,
                categories=categories)

        new_hits, new_preds, new_golds, new_srcs = seq2seq_utils.dump_outputs(
            pre_id.detach().cpu().numpy(), 
            post_out_id.detach().cpu().numpy(), 
            predicted_toks.detach().cpu().numpy(), 
            pre_tok_label_id.detach().cpu().numpy(), 
            id2tok, out_file)
        hits += new_hits
        preds += new_preds
        golds += new_golds
        srcs += new_srcs

    out_file.close()
    print(np.mean(hits))
    print(seq2seq_utils.get_bleu(preds, golds))

    writer.add_scalar('eval/bleu', seq2seq_utils.get_bleu(preds, golds), epoch + 1)
    writer.add_scalar('eval/true_hits', np.mean(hits), epoch+1)
