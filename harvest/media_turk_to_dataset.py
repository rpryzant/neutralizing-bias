import pandas as pd
import add_tags
from pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(s):
    return tokenizer.tokenize(s.strip())

input_file = '../../../data/persuasion/media_turk_data_annotated.csv'
df = pd.read_csv(input_file)
outputs = []
for i in range(len(df) // 3):
    responses = set([
        df.iloc[3 * i]['Answer.biased_word'],
        df.iloc[3 * i + 1]['Answer.biased_word'],
        df.iloc[3 * i + 2]['Answer.biased_word']])
    if len(responses) > 2:
        continue
    input = df.iloc[3 * i]['Input.text'].strip()
    tokenized_input = ' '.join(tokenize(input))
    pos_tags, dep_tags = add_tags.get_pos_dep(tokenize(input))
    outputs.append((df.iloc[3 * i]['Input.source'], tokenized_input,
                    tokenized_input, input, input, pos_tags, dep_tags))

output_file = '../../../data/persuasion/corpus.wordbiased.tag.ibc'
with open(output_file, 'w') as outfile:
    for output in outputs:
        outfile.write('\t'.join(output) + '\n')