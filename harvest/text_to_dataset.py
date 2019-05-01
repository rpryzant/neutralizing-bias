import add_tags
from pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(s):
    return tokenizer.tokenize(s.strip())

input_file = '../../../data/persuasion/possible options.txt'
outputs = []
with open(input_file, 'r') as infile:
    for line in infile:
        input = line.strip()
        tokenized_input = ' '.join(tokenize(input))
        pos_tags, dep_tags = add_tags.get_pos_dep(tokenize(input))
        outputs.append(('-1', tokenized_input, tokenized_input, input, input,
                        pos_tags, dep_tags))

output_file = '../../../data/persuasion/corpus.wordbiased.tag.infer'
with open(output_file, 'w') as outfile:
    for output in outputs:
        outfile.write('\t'.join(output) + '\n')
