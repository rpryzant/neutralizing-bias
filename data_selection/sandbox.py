from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm
import pickle

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
		if i > 10: continue
		parts = l.strip().split('\t')
		revID = parts[0]
		src_raw = parts[3]
		tgt_raw = parts[4]

		out['revID'].append(revID)
		out['src'].append(src_raw)
		out['tgt'].append(tgt_raw)

		src_emb = model.encode([src_raw])[0]
		tgt_emb = model.encode([tgt_raw])[0]

		out['src_emb'].append(src_emb)
		out['tgt_emb'].append(tgt_emb)

	return out

train_data = read_data(train)
dev_data = read_data(dev)
test_data = read_data(test)

pickle.dump(train_data, open('out.pkl', 'wb'))
pickle.dump(dev_data, open('out.pkl', 'wb'))
pickle.dump(test_data, open('out.pkl', 'wb'))
