'''
Comparison different bias vectors. Visualizing the difference for preliminary
analysis of bias labeling.
'''
# run minimal job:
# python seq2seq/train.py --train ../../data/v6/corpus.wordbiased.tag.train --test ../../data/v6/corpus.wordbiased.tag.test --working_dir TEST --max_seq_len --train_batch_size 3 --test_batch_size 10  --hidden_size 32 --debug_skip

import os
import shutil
import sys
from tqdm import tqdm
from simplediff import diff
from tensorboardX import SummaryWriter
import torch
from sklearn.mixture import BayesianGaussianMixture as BGM
from shared.args import ARGS
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer


embeddings = None
tok2id = None
SUBSET = 2000

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

    if os.path.exists(ARGS.working_dir + '/00000'):
        shutil.rmtree(ARGS.working_dir + '/00000')

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
    train_data_path = ARGS.train
    embedding_diff_list = []
    for i, line in enumerate(tqdm(open(train_data_path))):
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
        if i == SUBSET:
            break

    print("Number of datapoints: {}".format(SUBSET))
    embedding_diff_total = torch.stack(embedding_diff_list).squeeze()
    cluster_input = embedding_diff_total.detach().numpy()

    print("Creating bgm")
    bgm = BGM(n_components=4)
    labels = bgm.fit_predict(cluster_input)
    print("Writing out labels")

    with open("labels.tsv", "w+") as labels_file:
        for label in labels:
            labels_file.write("{} \n".format(label))

    '''
    Clustering and visualizating word embeddings.
    '''
    writer.add_embedding(embedding_diff_total)

    labels_predicted = []
    print("Printing out examples")

    for i, line in enumerate(tqdm(open(train_data_path))):
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
        embedding_diff = get_diff_embedding(words_diff).detach().numpy()
        predicted_label = bgm.predict(embedding_diff)
        if predicted_label not in labels_predicted:
            labels_predicted.append(predicted_label)
        print("Predicted Label: {}".format(predicted_label))
        print("pre: ")
        print(pre)
        print("post: ")
        print(post)
        print("\n \n ")
        keep_going = input("Continue? ") == 'y'
        if not keep_going:
            exit()


if __name__ == '__main__':
    main()
