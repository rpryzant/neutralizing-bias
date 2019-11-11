"""
usage
python parse_statement_quality_results.py Batch_3534942_batch_results.csv

"""
import sys
import csv
import numpy as np
from collections import Counter
out = []

with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    for row in reader:

        reversed = int(row['Input.reverse'])

        try:
            bias = [(row['Answer.bias_%d.on' % x] == 'true', x) for x in range(-2, 3)]
            bias = next( (d for t, d in bias if t ) )

            fluency = [(row['Answer.fluency_%d.on' % x] == 'true', x) for x in range(-2, 3)]
            fluency = next( (d for t, d in fluency if t ) )

            meaning = [(row['Answer.meaning_%d.on' % x] == 'true', x) for x in range(1, 4)]
            meaning = next( (d for t, d in meaning if t ) )
        except StopIteration:
            continue

        # [pre, pred]
        if reversed == 0:
            pre = row['Input.first_text']
            pred = row['Input.second_text']

        # [pred, pre]
        else:
            pred = row['Input.first_text']
            pre = row['Input.second_text']

            bias *= -1
            fluency *= -1

        if bias < 0:
            print('PRE:\t\t', pre)
            print('PREDICTED:\t', pred)
            print('bias:\t', bias)
            print()


        out.append([pre, pred, bias, fluency, meaning])

"""
print('avg bias: ', np.mean([x[2] for x in out]))
print('bias dist: ', Counter([x[2] for x in out]))
print()
print('avg fluency: ', np.mean([x[3] for x in out]))
print('fluency dist: ',  Counter([x[3] for x in out]))
print()
print('avg meaning: ', np.mean([x[4] for x in out]))
print('meaning dist: ',  Counter([x[4] for x in out]))
"""
