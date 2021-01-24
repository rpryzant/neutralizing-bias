"""
add tags to a corpusfile (output of gen_data_from_crawl.py)

"""
import sys
import spacy
from tqdm import tqdm
import codecs

NLP = spacy.load('en_core_web_sm')


def get_pos_dep(toks):
    def words_from_toks(toks):
        words = []
        word_indices = []
        for i, tok in enumerate(toks):
            if tok.startswith('##'):
                words[-1] += tok.replace('##', '')
                word_indices[-1].append(i)
            else:
                words.append(tok)
                word_indices.append([i])
        return words, word_indices

    out_pos, out_dep = [], []
    words, word_indices = words_from_toks(toks)
    analysis = NLP(' '.join(words))
    
    if len(analysis) != len(words):
        return None, None

    for analysis_tok, idx in zip(analysis, word_indices):
        out_pos += [analysis_tok.pos_] * len(idx)
        out_dep += [analysis_tok.dep_] * len(idx)
    
    assert len(out_pos) == len(out_dep) == len(toks)
    
    return ' '.join(out_pos), ' '.join(out_dep)
    

def main(in_file, out_file):
    out = codecs.open(out_file, 'w', 'utf=8')
    for line in tqdm(codecs.open(in_file, 'r', "utf-8"), total=sum(1 for _ in codecs.open(in_file, 'r', "utf-8"))):
    	parts = line.strip().split('\t')
    	if len(parts) != 5:
    		continue
    		
    	pre_pos, pre_dep = get_pos_dep(parts[1].split())
    	
    	if pre_pos is not None and pre_dep is not None:
    		out.write('\t'.join(parts + [pre_pos, pre_dep]))

if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    main(in_file, out_file)
