import re
import sys
import csv
import operator
import numpy as np
import string, pickle, os
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize

from util import *

printable = set(string.printable)

csv.field_size_limit(sys.maxsize)

# special characters
separator = 0
mask_char = 1 
unknown   = 2
to_TBD 	  = 3
offset 	  = 4

def wiki_text_clean(text):
	# text_ = filter(lambda x: x in printable, text)
	text_ = ''.join(filter(lambda x:x in string.printable, text))
	tokens = wordpunct_tokenize(text_)
	# tokens = [s.encode('utf-8') for s in tokens]
	return tokens 

def collect_revision_text(rev_ids):
	rev_size = len(rev_ids)
	cnt = 0
	X = {}

	for rev_id in rev_ids:
		print('processing revision id = ' + str(rev_id) + ', ' + str(cnt) + '/' + str(rev_size))

		url = 'https://en.wikipedia.org/wiki/?diff=' + str(rev_id)
		prevs_, nexts_ = url2diff(url)

		if len(prevs_) != len(nexts_):
			print('prev and next docs size not equal')
			print(len(prevs_))
			print(len(nexts_))
			exit(-1)

		cnt += 1
		prevs, nexts = [], []

		for i in range(len(prevs_)):
			prev_ = prevs_[i]
			prev_tokens = wiki_text_clean(prev_)
			if i == 0:
				prevs.extend(prev_tokens)
			else:
				nexts.extend([0] + prev_tokens)
		

		for i in range(len(nexts_)):
			next_ = nexts_[i]
			next_tokens = wiki_text_clean(next_)
		
			if i == 0:
				nexts.extend(next_tokens)
			else:
				nexts.extend([0] + next_tokens)

		X[rev_id] = (prevs, nexts)

	return X


def go(filename):
	# rev_id	rev_comment	rev_user	rev_user_text	rev_timestamp	rev_minor_edit
	with open(filename, 'r') as f:
		data = list(csv.reader(f, delimiter='\t'))

	rev_ids = [r[0] for r in data[1:]]

	X = collect_revision_text(rev_ids)
	pickle.dump(X, open(filename[:-3] + 'revision_text.pkl', 'wb'))

if __name__ == '__main__':
	go(sys.argv[1])




