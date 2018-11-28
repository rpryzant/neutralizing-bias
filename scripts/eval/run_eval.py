"""
run eval on a pair of corpus files

"""
import sys
sys.path.append('../../src/style_transfer_baseline')
import evaluation

src_path = sys.argv[1]
pred_path = sys.argv[2]
tgt_path = sys.argv[3]

src = [x.strip().split() for x in open(src_path)]
pred = [x.strip().split() for x in open(pred_path)]
tgt = [x.strip().split() for x in open(tgt_path)]

print(evaluation.get_metrics(src, pred, tgt))

