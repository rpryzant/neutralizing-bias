"""
python interpretation/feature_importance.py --checkpoint interpretation/models/inference_model_toksm_cov_4.ckpt


TODO DOUBLE CHECK!!
"""

import torch
import sys
import numpy as np
from collections import defaultdict

import sys; sys.path.append('.')
from shared.args import ARGS
from shared.data import REL2ID, POS2ID
from tagging.features import Featurizer




def importance_scores(ckpt_path):

	state_dict = torch.load(ckpt_path, map_location='cpu')

	l = []
	for key, matrix in state_dict.items():
		l += list(matrix.numpy().flatten())

	print('mean: ', np.mean(l))
	print('std: ', np.std(l))
	print()

	featurizer = Featurizer()
	feature_names = featurizer.get_feature_names()
	num_feats = len(feature_names)

	in_matrix = state_dict['tagging_model.tok_classifier.enricher.0.weight'].numpy()

	out_matrix = state_dict['tagging_model.tok_classifier.out.0.weight'].numpy()
	# only look at rows the features are multiplied by
	out_matrix = out_matrix[:, -num_feats:].transpose()

	def relu(x):
	    return x if x >= 0 else 0

	scores = defaultdict(lambda: defaultdict(int))


	for feature_i, feature in enumerate(feature_names):
	    for hidden_j in range(90):
	        in_weight = in_matrix[feature_i][hidden_j]
	        
	        for out_k in range(2):
	            out_weight = out_matrix[hidden_j][out_k]

	            scores[feature][out_k] += relu(in_weight) * out_weight


	positive_scores = {feature: scores[feature][1] for feature in feature_names}
	
	return positive_scores
	

if __name__ == '__main__':
	scores = importance_scores(ARGS.checkpoint)

	for feat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
	    print(feat, score)

