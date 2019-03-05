'''
Comparison different bias vectors. Visualizing the difference for preliminary
analysis of bias labeling.
'''
# run minimal job:
# python seq2seq/train.py --train ../../data/v6/corpus.wordbiased.tag.train --test ../../data/v6/corpus.wordbiased.tag.test --working_dir TEST --max_seq_len --train_batch_size 3 --test_batch_size 10  --hidden_size 32 --debug_skip

import os
import sys
from tqdm import tqdm
from simplediff import diff
from tensorboardX import SummaryWriter
import torch
from shared.args import ARGS
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projectors
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer


embeddings = None
tok2id = None

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
    diff_embedding = torch.zeros(1, 768)
    for word in words_diff:
        word_id = tok2id[word]
        input_indx = torch.LongTensor([word_id])
        diff_embedding += embeddings.word_embeddings(input_indx)
    diff_embedding /= len(words_diff)
    return diff_embedding

def setup():
    '''
    Setup initial folder settings, data storage and logging information. s
    '''
    if not os.path.exists(ARGS.working_dir):
        os.makedirs(ARGS.working_dir)

    with open(ARGS.working_dir + '/command.sh', 'w') as f:
        f.write('python' + ' '.join(sys.argv) + '\n')

def main():
    global embeddings, tok2id

    setup()
    writer = SummaryWriter(ARGS.working_dir)

    model = BertModel.from_pretrained(
        'bert-base-uncased',
        cache_dir=ARGS.working_dir + '/cache')
    embeddings = model.embeddings

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=ARGS.working_dir + '/cache')
    tok2id = tokenizer.vocab
    tok2id['<del>'] = len(tok2id)

    skipped = 0
    data_path = ARGS.train
    embedding_diff_list = []
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
        embedding_diff_list.append(embedding_diff)
    embedding_diff_total = torch.stack(embedding_diff_list).squeeze()

    '''
    Clustering and visualizating word embeddings.
    '''
    writer.add_embedding(embedding_diff_total)
    #Tensorflow Placeholders
    '''
    tf_embeddings = tf.convert_to_tensor(embedding_diff_total)
    diff_vocab_size = embedding_diff_total.shape[0]
    embedding_dim = embedding_diff_total.shape[1]

    X_init = tf.placeholder(tf.float32, shape=(diff_vocab_size, embedding_dim), name="embeddings")
    X = tf.Variable(X_init)

    #Initializer
    init = tf.global_variables_initializer()

    #Start Tensorflow Session
    sess = tf.Session()
    sess.run(init, feed_dict={X_init: tf_embeddings})

    #Instance of Saver, save the graph.
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(TENSORBOARD_FILES_PATH, sess.graph)
    '''

if __name__ == '__main__':
    main()
