'''
Cluster the underlying attention distribution to see if can detect
types of bias.
'''

import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GM
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import csv

window_size = 2
num_components = 2 # either 2,3 or 4

assert(num_components == 2) # currently only working with 2 components

results = pickle.load(open('attention_results.pkl','rb'))
survey_data = []
with open("bias_type_survey.tsv", 'r') as tsv_file:
    survey = csv.reader(tsv_file, delimiter='\t')
    header = next(survey)
    # 1:pre, 2:post, 3: epist, 4: framing, 5: prejudice, 6: noise
    for i,row in enumerate(survey):
        sentence = row[1]
        try:
            bias_classification = row[3:7].index('1')
        except:
            continue
        if bias_classification == 3:
            continue

        if bias_classification == 2:
            bias_classification = 1
        survey_data.append((sentence, bias_classification))


attention_vecs = []

outside_range_low = []
outside_range_high = []
for i,result in enumerate(results):
    words = result['input_toks']
    attention_dist = [float(num) for num in result['full_attention_dist']]
    len_entry = len(result['attention_dist'])

    attention_dist[len_entry] = 0 #watch out for this - setting punctuation to 0 attention
    attention_dist = window_size*[0] + attention_dist
    probs = result['probs']
    tok_labels = result['labels']

    bias_index = tok_labels.index(1)

    if(bias_index-window_size < 0):
        outside_range_low.append(i)
    elif(bias_index+window_size+1 > len_entry):
        outside_range_high.append(i)

    bias_index += window_size #because adding 0 padding before
    vec = attention_dist[bias_index-window_size:bias_index+1+window_size]
    attention_vecs.append(vec)

print("Number of entries past lower border: {}".format(len(outside_range_low)))
print("Number of entries past higher border: {}".format(len(outside_range_high)))

X = np.array(attention_vecs)
X /= np.expand_dims(np.amax(X, axis=1), axis= 1)

labels = GM(n_components=num_components).fit_predict(X)
print(labels)
X_embedded = PCA(n_components=2).fit_transform(X)

# printing out assignments
for component in range(num_components):
    indices = [i for i,l in enumerate(labels) if l == component]
    with open('examples_label_{}.txt'.format(component), 'w+') as f:
        for index in indices:
            f.write(" ".join(results[index]['input_toks']) + '\n \n')

# comparisons

predicted_labels = []
gt_labels = []
for sample in survey_data:
    sentence, label = sample
    sentence = sentence.split()
    for idx, result in enumerate(results):
        sample_sentence = result['input_toks']
        if(sentence[:2] == sample_sentence[:2]):
            predicted_labels.append(labels[idx])
            gt_labels.append(label)
            continue

predicted_labels = np.array(predicted_labels)
gt_labels = np.array(gt_labels)
print("V1")
print(confusion_matrix(gt_labels, predicted_labels))
print("V2")
print(confusion_matrix(gt_labels, -1 * (predicted_labels-1)))

print(gt_labels)
print(predicted_labels)
#visualization
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.scatter(vis_x, vis_y, c=labels)
plt.show()
