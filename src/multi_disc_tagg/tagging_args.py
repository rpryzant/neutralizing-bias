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


### DEFAULT STUFF FOR TRAINING
parser.add_argument("--num_bias_labels", default=2, type=int, help="dont touch")
parser.add_argument("--num_tok_labels", default=3, type=int, help="dont touch")
parser.add_argument("--bert_model", default='bert-base-uncased', help="dont touch")
parser.add_argument("--train_batch_size", default=32, type=int, help="dont touch")
parser.add_argument("--test_batch_size", default=16, type=int, help="dont touch")
parser.add_argument("--epochs", default=5, type=int, help="dont touch")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="dont touch")
parser.add_argument("--max_seq_len", default=70, type=int, help="dont touch")

### LOSS WEIGHTING
parser.add_argument(
    "--debias_weight", 
    default=1.0, 
    type=float, 
    help="weight for 1's on loss"
)



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

parser.add_argument("--drop_words", help="dummy to make it work with seq2seq_data.py. REFACTOR OUT!!", type=str, default=None)

ARGS = parser.parse_args()
