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
    "--checkpoint",
    help="model checkpoint to continue from (INFERENCE ONLY)",
    type=str, default=''
)
parser.add_argument(
    "--out_prefix",
    help="prefix for writing outputs",
    type=str, default=''
)
parser.add_argument("--bert_model", 
    default='bert-base-uncased', 
    help="dont touch")
parser.add_argument("--train_batch_size", 
    default=32, type=int, 
    help="batch size")
parser.add_argument("--test_batch_size", 
    default=16, type=int, 
    help="batch size")


##################################################################################
##################################################################################
#                      JOINT ARGS
##################################################################################
##################################################################################
parser.add_argument(
    "--token_softmax",
    action='store_true',
    help='softmax over token dimension')
parser.add_argument(
    "--sequence_softmax",
    action='store_true',
    help='softmax over time dimension instead of token dist')
parser.add_argument(
    "--zero_threshold",
    type=float, default=10000.0,
    help='threshold for zeroing-out token scores')






##################################################################################
##################################################################################
#                      TAGGER ARGS
##################################################################################
##################################################################################




parser.add_argument("--num_tok_labels", 
    default=3, type=int, 
    help="dont touch")





#### Category stuff
parser.add_argument(
    "--train_categories_file",
    type=str, default=None,
    help='pointer to wikipedia categories')
parser.add_argument(
    "--test_categories_file",
    type=str, default=None,
    help='pointer to wikipedia categories')
parser.add_argument(
    "--argmax_categories",
    action='store_true',
    help='replace distribution with one-hot')
parser.add_argument(
    "--category_threshold",
    default=0.0, type=float,
    help='threshold ones')
parser.add_argument(
    "--predict_categories",
    action='store_true',
    help='use [CLS] to predict categories')
parser.add_argument(
    "--concat_categories",
    action='store_true',
    help='concat raw category vec to help tag')
parser.add_argument(
    "--category_emb",
    action='store_true',
    help='concat category embedding vec to help tag (use with --concat_categories) ')
parser.add_argument(
    "--category_input",
    action='store_true',
    help='prepend special category input emb to seq')
parser.add_argument(
    "--num_categories", 
    default=43, type=int, 
    help="number of categories (don't change!)")


### EXTRA FEATURE COMBINE FN
parser.add_argument(
    "--extra_features_method",
    help="how to add extra features: [concat, add]",
    default='concat'
)
parser.add_argument(
    "--combiner_layers",
    type=int, default=1, help="num layers for combiner: [1, 2]"
)
parser.add_argument(
    "--pre_enrich",
    help="pass features through linear layer before combination",
    action='store_true'
)
parser.add_argument(
    "--activation_hidden",
    help="activation functions on hidden layers",
    action='store_true'
)
### EXTRA FEATURES TOP!!
parser.add_argument(
    "--extra_features_top",
    help="add extra features by concating on top",
    action='store_true'
)
parser.add_argument(
    "--small_waist",
    help="make hidden layer for 2-layer combiner the smaller of the inputs (for 2-layer combiners)",
    action='store_true'
)
parser.add_argument(
    "--lexicon_feature_bits",
    type=int, default=1, help="num bits for lexicon features: [1, 2]"
)
### EXTRA FEATURES BOTTOM!!
parser.add_argument(
    "--extra_features_bottom",
    help="add extra features by concating on bottom",
    action='store_true'
)
parser.add_argument("--share_combiners", 
	help="share parameters if multiple combiners", action='store_true')

parser.add_argument("--combine1", help="combine location 1", action='store_true')
parser.add_argument("--combine2", help="combine location 2", action='store_true')
parser.add_argument("--combine3", help="combine location 3", action='store_true')
parser.add_argument("--combine4", help="combine location 4", action='store_true')
parser.add_argument("--combine5", help="combine location 5", action='store_true')
parser.add_argument("--combine6", help="combine location 6", action='store_true')















##################################################################################
##################################################################################
#                      SEQ2SEQ ARGS
##################################################################################
##################################################################################




# seq2seq args
parser.add_argument(
    "--beam_width",
    help="beam width for evaluation (1 = greedy)",
    type=int, default=1
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



# taking tagging output options
parser.add_argument(
    "--tok_dist_train_path",
    help="token distributions to use",
    type=str, default=None
)
parser.add_argument(
    "--tok_dist_test_path",
    help="token distributions to use",
    type=str, default=None
)
parser.add_argument(
    "--tok_dist_softmax_prob",
    help="prob of passing tok_dist through softmax",
    type=float, default=0.0
)
parser.add_argument(
    "--tok_dist_mix_prob",
    help="prob of giving tok_dist instead of true tok labels (0 = all tok dist, 1 = no tok dist)",
    type=float, default=0.0
)
parser.add_argument(
    "--tok_dist_noise_prob",
    help="prob of replacing tok_dist with noisy one-hot vector",
    type=float, default=0.0
)
parser.add_argument(
    "--tok_dist_argmax",
    help="replace tok dist with bone-hot based on argmax",
    action='store_true'
)
parser.add_argument(
    "--tok_dist_threshold",
    help="threshold to replace probs with 1s on tok dist",
    type=float, default=0
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
parser.add_argument(
    "--ignore_pretrain_enrich",
    help="ignore enrichment during pretraining",
    action='store_true'
)



# loss settings
parser.add_argument(
    "--debias_weight",
    help="multiplyer for new words on target side loss",
    type=float, default=1.0

)
ARGS = parser.parse_args()
