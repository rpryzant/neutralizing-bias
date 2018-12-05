import sys

import json
import data
import models
import utils
import numpy as np
import logging
import argparse
import os
import time
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import evaluation
from cuda import CUDA



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument(
    "--bleu",
    help="do BLEU eval",
    action='store_true'
)
parser.add_argument(
    "--overfit",
    help="train continuously on one batch of data",
    action='store_true'
)
parser.add_argument(
    "--test",
    help="skip training and only test",
    action='store_true'
)
args = parser.parse_args()
config = json.load(open(args.config, 'r'))


working_dir = config['data']['working_dir']

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

config_path = os.path.join(working_dir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/train_log' % working_dir,
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info('Reading data ...')
src, tgt = data.read_nmt_data(
    src=config['data']['src'],
    config=config,
    tgt=config['data']['tgt']
)

src_test, tgt_test = data.read_nmt_data(
    src=config['data']['src_test'],
    config=config,
    tgt=config['data']['tgt_test'],
    train_src=src,
    train_tgt=tgt
)
logging.info('...done!')


batch_size = config['data']['batch_size']
max_length = config['data']['max_len']
src_vocab_size = len(src['tok2id'])
tgt_vocab_size = len(tgt['tok2id'])


weight_mask = torch.ones(tgt_vocab_size)
weight_mask[tgt['tok2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
if CUDA:
    weight_mask = weight_mask.cuda()
    loss_criterion = loss_criterion.cuda()

torch.manual_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])

model = models.SeqModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    pad_id_src=src['tok2id']['<pad>'],
    pad_id_tgt=tgt['tok2id']['<pad>'],
    config=config
)

logging.info('MODEL HAS %s params' %  model.count_params())
model, start_epoch = models.attempt_load_model(
    model=model,
    checkpoint_dir=working_dir)
if CUDA:
    model = model.cuda()

writer = SummaryWriter(working_dir)


lr = config['training']['learning_rate']
if config['training']['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

epoch_loss = []
start_since_last_report = time.time()
words_since_last_report = 0
gen_losses_since_last_report = []
side_losses_since_last_report = []
best_metric = 0.0
best_epoch = 0
cur_metric = 0.0 # log perplexity or BLEU
num_batches = len(src['content']) / batch_size
with open(working_dir + '/stats_labels.csv', 'w') as f:
    f.write(utils.config_key_string(config) + ',%s,%s\n' % (
        ('bleu' if args.bleu else 'dev_loss'), 'best_epoch'))

STEP = 0
for epoch in range(start_epoch, config['training']['epochs']):
    # if epoch > 3 and cur_metric == 0 or epoch > 7 and cur_metric < 10 or epoch > 15 and cur_metric < 15:
    #     logging.info('QUITTING...NOT LEARNING WELL')
    #     with open(working_dir + '/stats.csv', 'w') as f:
    #         f.write(utils.config_val_string(config) + ',%s,%s\n' % (
    #             best_metric, best_epoch))
    #     break

    if cur_metric > best_metric:
        # rm old checkpoint
        for ckpt_path in glob.glob(working_dir + '/model.*'):
            os.system("rm %s" % ckpt_path)
        # replace with new checkpoint
        torch.save(model.state_dict(), working_dir + '/model.%s.ckpt' % epoch)

        best_metric = cur_metric
        best_epoch = epoch - 1

    losses = []
    for i in range(0, len(src['data']), batch_size):
        if args.test:
            continue
        if args.overfit:
            i = 0

        batch_idx = i / batch_size

        input_content, input_aux, output, side_info, _ = data.minibatch(
            src, tgt, i, batch_size, max_length, config)
        input_lines_src, _, srclens, srcmask, _ = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output
        side_info, _, _, _, _ = side_info

        decoder_logit, decoder_probs, side_logit, side_loss = model(
            input_lines_src, input_lines_tgt, srcmask, srclens,
            input_ids_aux, auxlens, auxmask, side_info)

        optimizer.zero_grad()


        generator_loss = loss_criterion(
            decoder_logit.contiguous().view(-1, tgt_vocab_size),
            output_lines_tgt.view(-1)
        )
        loss = generator_loss + (side_loss * config['experimental']['side_loss_multiplyer'])

        losses.append(loss.data[0])
        gen_losses_since_last_report.append(generator_loss.data[0])
        if isinstance(side_loss, float):
            side_losses_since_last_report.append(side_loss)
        else:
            side_losses_since_last_report.append(side_loss.data[0])
        epoch_loss.append(loss.data[0])
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])

        writer.add_scalar('stats/grad_norm', norm, STEP)

        optimizer.step()

        if args.overfit or batch_idx % config['training']['batches_per_report'] == 0:

            s = float(time.time() - start_since_last_report)
            wps = (batch_size * config['training']['batches_per_report']) / s
            avg_gen_loss = np.mean(gen_losses_since_last_report)
            avg_side_loss = np.mean(side_losses_since_last_report)
            info = (epoch, batch_idx, num_batches, wps, avg_gen_loss + avg_side_loss, cur_metric)
            writer.add_scalar('stats/WPS', wps, STEP)
            writer.add_scalar('stats/gen_loss', avg_gen_loss, STEP)
            writer.add_scalar('stats/side_loss', avg_side_loss, STEP)
            logging.info('EPOCH: %s ITER: %s/%s WPS: %.2f LOSS: %.4f METRIC: %.4f' % info)
            start_since_last_report = time.time()
            words_since_last_report = 0
            gen_losses_since_last_report = []
            side_losses_since_last_report = []

        # NO SAMPLING!! because weird train-vs-test data stuff would be a pain
        STEP += 1
    if args.overfit:
        continue

    logging.info('EPOCH %s COMPLETE. EVALUATING...' % epoch)
    start = time.time()
    model.eval()
    dev_loss = evaluation.evaluate_lpp(
            model, src_test, tgt_test, config)

    writer.add_scalar('eval/loss', dev_loss, epoch)

    if args.bleu and epoch >= config['training'].get('bleu_start_epoch', 1):
        metrics, inputs, preds, golds, auxs = evaluation.inference_metrics(
            model, src_test, tgt_test, config)

        with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(working_dir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(working_dir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')

        writer.add_scalar('eval/tgt_precision', metrics['tgt_precision'], epoch)
        writer.add_scalar('eval/tgt_recall', metrics['tgt_recall'], epoch)
        writer.add_scalar('eval/src_precision', metrics['src_precision'], epoch)
        writer.add_scalar('eval/src_recall', metrics['src_recall'], epoch)
        writer.add_scalar('eval/edit_distance', metrics['edit_distance'], epoch)
        writer.add_scalar('eval/bleu', metrics['bleu'], epoch)
        writer.add_scalar('eval/src_bleu', metrics['src_bleu'], epoch)
        writer.add_scalar('eval/tgt_bleu', metrics['tgt_bleu'], epoch)
        writer.add_scalar('eval/classifier_error', metrics['classifier_error'], epoch)

        cur_metric = metrics['bleu']

    else:
        cur_metric = dev_loss

    model.train()

    logging.info('METRIC: %s. TIME: %.2fs CHECKPOINTING...' % (
        cur_metric, (time.time() - start)))
    avg_loss = np.mean(epoch_loss)
    epoch_loss = []
    
writer.close()
with open(working_dir + '/stats.csv', 'w') as f:
    f.write(utils.config_val_string(config) + ',%s,%s\n' % (
        best_metric, best_epoch))

