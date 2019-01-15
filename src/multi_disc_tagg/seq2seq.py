# python seq2seq.py --train ../../data/v5/final/bias --test ../../data/v5/final/bias --working_dir TEST/

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import os
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from simplediff import diff
import pickle
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import Counter
import math

from pytorch_pretrained_bert.modeling import BertEmbeddings
from pytorch_pretrained_bert.optimization import BertAdam

from data import get_dataloader
import seq2seq_model
from args import ARGS


BERT_MODEL = "bert-base-uncased"

train_data_prefix = ARGS.train
test_data_prefix = ARGS.test

working_dir = ARGS.working_dir
if not os.path.exists(working_dir):
    os.makedirs(working_dir)


TRAIN_TEXT = train_data_prefix + '.train.pre'
TRAIN_TEXT_POST = train_data_prefix + '.train.post'

TEST_TEXT = test_data_prefix + '.test.pre'
TEST_TEXT_POST = test_data_prefix + '.test.post'

WORKING_DIR = working_dir

NUM_BIAS_LABELS = 2
NUM_TOK_LABELS = 3

if ARGS.bert_encoder:
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
else:
    TRAIN_BATCH_SIZE = 80
    TEST_BATCH_SIZE = 80

EPOCHS = 60

MAX_SEQ_LEN = 70

CUDA = (torch.cuda.device_count() > 0)
                                                                
def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
        return 100 * bleu(stats)



def train_for_epoch(model, dataloader, tok2id, optimizer, criterion):
    global CUDA

    losses = []
    for step, batch in enumerate(tqdm(dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        pre_id, pre_mask, pre_len, post_in_id, post_out_id, tok_label_id, replace_id, _, _ = batch

        post_logits, post_probs = model(pre_id, post_in_id, pre_mask, pre_len, tok_label_id)
        loss = criterion(post_logits.contiguous().view(-1, len(tok2id)), post_out_id.contiguous().view(-1))

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        model.zero_grad()

        losses.append(loss.detach().cpu().numpy())

    return losses


def dump_outputs(src_ids, gold_ids, predicted_ids, gold_replace_id, gold_tok_dist, id2tok, out_file):
    out_hits = []
    preds_for_bleu = []
    golds_for_bleu = []

    for src_seq, gold_seq, pred_seq, gold_replace, gold_dist in zip(
        src_ids, gold_ids, predicted_ids, gold_replace_id, gold_tok_dist):

        src_seq = [id2tok[x] for x in src_seq]
        gold_seq = [id2tok[x] for x in gold_seq]
        pred_seq = [id2tok[x] for x in pred_seq[1:]]
        gold_seq = gold_seq[:gold_seq.index('止')]
        if '止' in pred_seq:
            pred_seq = pred_seq[:pred_seq.index('止')]
        src_seq = ' '.join(src_seq).replace('[PAD]', '').strip()
        gold_seq = ' '.join(gold_seq).replace('[PAD]', '').strip()
        pred_seq = ' '.join(pred_seq).replace('[PAD]', '').strip()

        gold_replace = id2tok[gold_replace]
        pred_replace = [chunk for tag, chunk in diff(src_seq.split(), pred_seq.split()) if tag == '+']
        try:
            print('#' * 80, file=out_file)
            print('IN SEQ: \t', src_seq.encode('utf-8'), file=out_file)
            print('GOLD SEQ: \t', gold_seq.encode('utf-8'), file=out_file)
            print('PRED SEQ:\t', pred_seq.encode('utf-8'), file=out_file)
            print('GOLD DIST: \t', list(gold_dist), file=out_file)
            print('GOLD TOK: \t', gold_replace.encode('utf-8'), file=out_file)
            print('PRED TOK: \t', pred_replace, file=out_file)
        except UnicodeEncodeError:
            pass

        if gold_seq == pred_seq:
            out_hits.append(1)
        else:
            out_hits.append(0)

        preds_for_bleu.append(pred_seq.split())
        golds_for_bleu.append(gold_seq.split())

    return out_hits, preds_for_bleu, golds_for_bleu


def run_eval(model, dataloader, tok2id, out_file_path):
    global MAX_SEQ_LEN
    global CUDA

    id2tok = {x: tok for (tok, x) in tok2id.items()}

    weight_mask = torch.ones(len(tok2id))
    weight_mask[0] = 0
    criterion = nn.CrossEntropyLoss(weight=weight_mask)

    out_file = open(out_file_path, 'w')

    losses = []
    hits = []
    preds, golds = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        pre_id, pre_mask, pre_len, post_in_id, post_out_id, tok_label_id, replace_id, _, _ = batch
        
        post_start_id = tok2id['[CLS]']
        max_len = min(MAX_SEQ_LEN, pre_len[0].detach().cpu().numpy() + 10)

        with torch.no_grad():
            predicted_toks = model.inference_forward(
                pre_id, post_start_id, pre_mask, pre_len, max_len, tok_label_id,
                beam_width=5)
#            loss = criterion(post_logits.contiguous().view(-1, len(tok2id)), post_out_id.contiguous().view(-1))

#        losses.append(loss.detach().cpu().numpy())
        new_hits, new_preds, new_golds = dump_outputs(
            pre_id.detach().cpu().numpy(), 
            post_out_id.detach().cpu().numpy(), 
            predicted_toks, 
            replace_id.detach().cpu().numpy(), 
            tok_label_id.detach().cpu().numpy(), 
            id2tok, out_file)
        hits += new_hits
        preds += new_preds
        golds += new_golds
    out_file.close()

    return [-1], hits, preds, golds






# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


if ARGS.pretrain_data: 
    pretrain_dataloader, num_pretrain_examples = get_dataloader(
        ARGS.pretrain_data, ARGS.pretrain_data, 
        tok2id, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/pretrain_data.pkl',
        noise=True)

train_dataloader, num_train_examples = get_dataloader(
    TRAIN_TEXT, TRAIN_TEXT_POST, 
    tok2id, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/train_data.pkl',
    add_del_tok=ARGS.add_del_tok)
eval_dataloader, num_eval_examples = get_dataloader(
    TEST_TEXT, TEST_TEXT_POST,
    tok2id, TEST_BATCH_SIZE, MAX_SEQ_LEN, WORKING_DIR + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok)



# # # # # # # # ## # # # ## # # MODELS # # # # # # # # ## # # # ## # #
if ARGS.no_tok_enrich:
    model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
else:
    model = seq2seq_model.Seq2SeqEnrich(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('NUM PARAMS: ', params)



# # # # # # # # ## # # # ## # # OPTIMIZERS, ETC # # # # # # # # ## # # # ## # #
writer = SummaryWriter(WORKING_DIR)

if ARGS.bert_encoder:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    num_train_steps = (num_train_examples * 40)
    if ARGS.pretrain_data: 
        num_train_steps += (num_pretrain_examples * ARGS.pretrain_epochs)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-5,
                         warmup=0.1,
                         t_total=num_train_steps)

else:
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

weight_mask = torch.ones(len(tok2id))
weight_mask[0] = 0
criterion = nn.CrossEntropyLoss(weight=weight_mask)

if CUDA:
    weight_mask = weight_mask.cuda()
    criterion = criterion.cuda()
    model = model.cuda()



# # # # # # # # # # # PRETRAINING (optional) # # # # # # # # # # # # # # # #
if ARGS.pretrain_data:
    print('PRETRAINING...')
    for epoch in range(ARGS.pretrain_epochs):
        model.train()
        losses = train_for_epoch(model, pretrain_dataloader, tok2id, optimizer, criterion)
        writer.add_scalar('pretrain/loss', np.mean(losses), epoch)



# # # # # # # # # # # # TRAINING # # # # # # # # # # # # # #
print('INITIAL EVAL...')
model.eval()
losses, hits, preds, golds = run_eval(model, eval_dataloader, tok2id, WORKING_DIR + '/results_initial.txt')
writer.add_scalar('eval/bleu', get_bleu(preds, golds), 0)
writer.add_scalar('eval/true_hits', np.mean(hits), 0)



for epoch in range(EPOCHS):
    print('EPOCH ', epoch)
    print('TRAIN...')
    model.train()
    losses = train_for_epoch(model, train_dataloader, tok2id, optimizer, criterion)
    writer.add_scalar('train/loss', np.mean(losses), epoch+1)
    print('EVAL...')

    model.eval()
    losses, hits, preds, golds = run_eval(model, eval_dataloader, tok2id, WORKING_DIR + '/results_%d.txt' % epoch)
    writer.add_scalar('eval/bleu', get_bleu(preds, golds), epoch+1)
#    writer.add_scalar('eval/loss', np.mean(losses), epoch)
    writer.add_scalar('eval/true_hits', np.mean(hits), epoch+1)




































