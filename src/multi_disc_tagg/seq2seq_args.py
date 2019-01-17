import argparse



parser = argparse.ArgumentParser()
# seq2seq args
parser.add_argument(
    "--train",
    help="train prefix",
)
parser.add_argument(
    "--test",
    help="test prefix",
    required=True
)
parser.add_argument(
    "--working_dir",
    help="train continuously on one batch of data",
    type=str, required=True
)
parser.add_argument(
    "--beam_width",
    help="beam width for evaluation (1 = greedy)",
    type=int, default=1
)
parser.add_argument(
    "--batch_size",
    help="batch size. IF BERT ENCODER SET THIS TO 16",
    type=int, default=80
)
parser.add_argument(
    "--epochs",
    help="training epochs",
    type=int, default=40
)
parser.add_argument(
    "--max_seq_len",
    type=int, default=70
)
parser.add_argument(
    "--hidden_size",
    help="hidden size of encoder/decoder",
    type=int, default=256
)

# inference only
parser.add_argument(
    "--checkpoint",
    help="model checkpoint to continue from",
    type=str, default=''
)
parser.add_argument(
    "--token_distribution_file",
    help="token distributions to use",
    type=str, default=''
)


# bert settings
parser.add_argument(
    "--bert_word_embeddings",
    help="use bert pretrained word embeddings",
    action='store_true'
)
parser.add_argument(
    "--bert_full_embeddings",
    help="use bert pretrained pos embeddings",
    action='store_true'
)
parser.add_argument(
    "--freeze_embeddings",
    help="freeze pretrained embeddings",
    action='store_true'
)
parser.add_argument(
    "--bert_encoder",
    help="freeze pretrained embeddings",
    action='store_true'
)

# debias settings
parser.add_argument(
    "--no_tok_enrich",
    help="turn off src enrichment",
    action='store_true'
)
parser.add_argument(
    "--add_del_tok",
    help="add a <del> tok for deletions",
    action='store_true'
)
parser.add_argument(
    "--fine_enrichment",
    help="seperate enrichment embedidings for del/edit",
    action='store_true'
)


# pretrain settings
parser.add_argument(
    "--pretrain_data",
    help="dataset for pretraining. NOT A PREFIX!!",
    type=str, default=''
)
parser.add_argument(
    "--pretrain_epochs",
    help="dataset for pretraining. NOT A PREFIX!!",
    type=int, default=4
)
parser.add_argument(
    "--noise_prob",
    help="drop prob for noising",
    type=float, default=0.25
)
parser.add_argument(
    "--shuf_dist",
    help="local shuffle distance (-1 for global shuffle, 0 for no shuffle, 1+ for local)",
    type=int, default=3
)
parser.add_argument(
    "--drop_words",
    help="list of words to drop",
    type=str, default=None
)
parser.add_argument(
    "--keep_bigrams",
    help="keep bigrams together that occured in original when shuffling",
    action='store_true'
)



# loss settings
parser.add_argument(
    "--debias_weight",
    help="multiplyer for new words on target side loss",
    type=float, default=1.0

)
ARGS = parser.parse_args()
