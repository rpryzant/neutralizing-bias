"""
runs multi-bleu.perl on all of the prediction files in a directory
"""
import sys
import os
import glob
import re

def get_bleu(multi_bleu_output):
    bleu_re = 'BLEU = (\d{1,2}(\.\d*)?)'
    return re.match(bleu_re, multi_bleu_output).group(1)

dir = sys.argv[1]

pred_files = sorted(glob.glob(dir + '/preds.*'))

for pred_file in pred_files:
    gold_file = pred_file.replace('preds', 'golds')

    result = os.popen('perl multi-bleu.perl %s < %s 2> perl_junk' % (
        gold_file, pred_file)).read()
    print pred_file, get_bleu(result)

os.system('rm perl_junk')
