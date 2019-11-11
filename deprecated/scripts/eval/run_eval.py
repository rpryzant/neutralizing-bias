"""
run eval on a pair of corpus files

"""
import sys
sys.path.append('../../src/style_transfer_baseline')
import evaluation
import models

src_path = sys.argv[1]
pred_path = sys.argv[2]
tgt_path = sys.argv[3]
classifier_path = "../../data/v2/eval_classifier"

eval_classifier = models.TextClassifier.from_pickle(
    "../../data/v2/eval_classifier")


src = [x.strip().split() for x in open(src_path)]
pred = [x.strip().split() for x in open(pred_path)]
tgt = [x.strip().split() for x in open(tgt_path)]

print(evaluation.get_metrics(src, pred, tgt, classifier=eval_classifier))

