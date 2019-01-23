"""
make fake rel and pos file because my code is shit

 python fake_rel_pos.py ../../data/v5/test_joined_testset/decoding.train.pre 
 python fake_rel_pos.py ../../data/v5/test_joined_testset/decoding.test.pre 
"""
import sys

in_f = sys.argv[1]


out_rel = open(in_f + '.rel', 'w')
out_pos = open(in_f + '.pos', 'w')

for l in open(in_f):
    l = l.strip().split()
    out_rel.write(' '.join(['<UNK>'] * len(l)) + '\n')
    out_pos.write(' '.join(['<UNK>'] * len(l)) + '\n')
