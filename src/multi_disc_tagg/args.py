import argparse



parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    help="train prefix",
    required=True
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
    "--hidden_size",
    help="hidden size of encoder/decoder",
    type=int, default=256
)
ARGS = parser.parse_args()