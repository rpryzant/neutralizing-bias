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

ARGS = parser.parse_args()
