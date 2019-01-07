"""
pull pretrained word vectors out of the bert model

"""
import os
from tqdm import tqdm
import numpy as np

import torch

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

WORKING_DIR = "tmp"
BERT_MODEL = "bert-base-uncased"


tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=WORKING_DIR + '/cache')
model = BertModel.from_pretrained(
    BERT_MODEL,
    WORKING_DIR + '/cache')

embeddings = model.embeddings.word_embeddings


def cos_dist(word1, word2):
    id1 = tokenizer.convert_tokens_to_ids([word1]) 
    id2 = tokenizer.convert_tokens_to_ids([word2]) 

    embeddings = model.embeddings.word_embeddings(torch.LongTensor([id1, id2])).squeeze(1)

    return torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)


for (tok, idx) in tokenizer.vocab.items():

    vec = model.embeddings.word_embeddings(torch.LongTensor([idx])).squeeze()
    print(tok + ' ' + ' '.join(str(x) for x in vec.detach().numpy().tolist()))

print('<del>' + ' ' + ' '.join(str(x) for x in np.random.uniform(low=-0.1, high=0.1, size=768).tolist()))
