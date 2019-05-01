import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from collections import Counter

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(s):
    return tokenizer.tokenize(s.strip())

outputs = {}
pred_file = '../../../data/persuasion/ibc_output.txt'
with open(pred_file) as f:
    for line in f:
        if not line.startswith('Sentence:'):
            continue
        line = line.replace('Sentence: ', '')
        line = line.replace(' [PAD]', '')
        line = line.strip()
        next(f)
        pred = next(f)
        assert pred.startswith('Prediction: ')
        pred = pred.replace('Prediction: ', '')
        pred = pred.strip()
        if line in outputs:
            print(line)
            print("!!!!!")
        outputs[line] = pred
turk_file = '../../../data/persuasion/media_turk_data_annotated.csv'
df = pd.read_csv(turk_file)
correct = 0
total = 0
for i in range(len(df) // 3):
    input = df.iloc[3 * i]['Input.text'].strip()
    tokenized_input = ' '.join(tokenize(input))
    if tokenized_input not in outputs:
        continue
    total += 1
    responses = Counter()
    responses[df.iloc[3 * i]['Answer.biased_word'].lower()] += 1
    responses[df.iloc[3 * i + 1]['Answer.biased_word'].lower()] += 1
    responses[df.iloc[3 * i + 2]['Answer.biased_word'].lower()] += 1
    target = responses.most_common(1)[0][0]
    if target == outputs[tokenized_input]:
        correct += 1
    del outputs[tokenized_input]

acc = correct / total
print(acc)
