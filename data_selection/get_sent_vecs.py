from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm
import pickle
from simplediff import diff

data_prefix = 'bias_data/WNC/biased.word'

train = data_prefix + '.train'
dev = data_prefix + '.dev'
test = data_prefix + '.test'

model = SentenceTransformer('bert-base-nli-mean-tokens')


def wc_l(path):
	return sum(1 for _ in open(path))

def read_data(path):
	out = defaultdict(list)

	for i, l in tqdm(enumerate(open(path)), total=wc_l(path)):
		# if i > 10: continue
		parts = l.strip().split('\t')
		revID = parts[0]
		src_raw = parts[3]
		tgt_raw = parts[4]

		out['revID'].append(revID)
		out['src'].append(src_raw)
		out['tgt'].append(tgt_raw)

		src_emb = model.encode([src_raw])[0]
		out['src_emb'].append(src_emb)

		tgt_emb = model.encode([tgt_raw])[0]
		out['tgt_emb'].append(tgt_emb)

		s_diff = diff(src_raw.split(), tgt_raw.split())
		shared = []
		src_unique = []
		tgt_unique = []
		for tag, chunk in s_diff:
			if tag == '=':
				shared += chunk
			elif tag == '+':
				tgt_unique += chunk
			elif tag == '-':
				src_unique += chunk

		shared = ' '.join(shared)
		shared_emb = model.encode([shared])[0]
		out['shared'].append(shared)
		out['shared_emb'].append(shared_emb)

		tgt_unique = ' '.join(tgt_unique)
		tgt_unique_emb = model.encode([tgt_unique])[0]
		out['tgt_unique'].append(tgt_unique)
		out['tgt_unique_emb'].append(tgt_unique_emb)

		print(tgt_unique)
		# print(tgt_unique_emb)

		src_unique = ' '.join(src_unique)
		src_unique_emb = model.encode([src_unique])[0]
		out['src_unique'].append(src_unique)
		out['src_unique_emb'].append(src_unique_emb)

	return out


train_data = read_data(train)
dev_data = read_data(dev)
test_data = read_data(test)

pickle.dump(train_data, open('out.train.pkl', 'wb'))
pickle.dump(dev_data, open('out.dev.pkl', 'wb'))
pickle.dump(test_data, open('out.test.pkl', 'wb'))
