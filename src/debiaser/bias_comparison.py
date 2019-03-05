'''
Comparison different bias vectors. Visualizing the difference for preliminary
analysis of bias labeling.
'''
# run minimal job:
# python seq2seq/train.py --train ../../data/v6/corpus.wordbiased.tag.train --test ../../data/v6/corpus.wordbiased.tag.test --working_dir TEST --max_seq_len --train_batch_size 3 --test_batch_size 10  --hidden_size 32 --debug_skip

from tqdm import tqdm
import click
from simplediff import diff
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

def get_difference_words(diff):
    '''
    Returns the difference words in one list.
    args:
        *diff ([(String, [String])]): List of tuples where the first entry
            represents the chaged action between the different words, and the
            second entry represents the different words.
    returns:
        *diff_list ([String]): Words that are only different between the two
            sentences
    '''
    diff_list = []
    for (comparison, words) in diff:
        if comparison != '=':
            diff_list.extend(words)
    return diff_list

def get_diff_embedding(words_diff):
    '''
    Returns word embeddings of the different words.
    '''
    

@click.command()
@click.argument('data_path')
def main(data_path):
    skipped = 0
    for i, line in enumerate(tqdm(open(data_path))):
        parts = line.strip().split('\t')

        # if there pos/rel info
        if len(parts) == 7:
            [revid, pre, post, _, _, _, _] = parts
        # no pos/rel info
        elif len(parts) == 5:
            [revid, pre, post, _, _] = parts
        # broken line
        else:
            skipped += 1
            continue

        tokens = pre.strip().split()
        post_tokens = post.strip().split()
        tok_diff = diff(tokens, post_tokens)
        words_diff = get_difference_words(tok_diff)
        embedding_diff = get_diff_embedding(words_diff)

        # Create difference embedding from mean of word embeddings in words_diff

        # Custering word embeddings


if __name__ == '__main__':
    main()
