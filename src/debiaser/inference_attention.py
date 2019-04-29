'''
Loads in the bias tagging model and attempt to analyze the attention
distributions that are produced by the attention layers of the model.
'''

# model import statements
from shared.args import ARGS
from shared.constants import CUDA
from shared.data import get_dataloader

import math
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import joint.model as joint_model

import pickle

# # # # # # # # ## # # # ## # # BERT INITIALIZATION # # # # # # # # ## # # # ## # #

tokenizer = BertTokenizer.from_pretrained(
    ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab

tok2id['<del>'] = len(tok2id)
print("The len of our vocabulary is {}".format(len(tok2id)))

# # # # # # # # ## # # # ## # # TAGGING MODEL # # # # # # # # ## # # # ## # #

if ARGS.extra_features_top:
    tag_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    tag_model = tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    tag_model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache')
if CUDA:
    tag_model = tag_model.cuda()

# # # # # # # # ## # # # ## # # DEBIAS MODEL # # # # # # # # ## # # # ## # #
# bulid model
if ARGS.pointer_generator:
    debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id) # 768 = bert hidden size
else:
    debias_model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
if CUDA:
    debias_model = debias_model.cuda()


# # # # # # # # ## # # # ## # # LOADING CHECKPOINT # # # # # # # # ## # # # ## # #

checkpoint = torch.load('joint/final_model/model_7.ckpt')
joint_model = joint_model.JointModel(debias_model=debias_model, tagging_model=tag_model)
joint_model.load_state_dict(checkpoint)


eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, ARGS.test_batch_size, ARGS.working_dir + '/test_data.pkl',
    categories_path=ARGS.categories_file,
    test=True, add_del_tok=ARGS.add_del_tok)

joint_model.eval()
print('loaded in model')



def transpose_for_scores(x):
    '''
    This is used to extract the final attention distribution
    '''
    num_attention_heads = 1
    attention_head_size = 768
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

def run_attention_extractor():
    global results
    for name, module in joint_model.named_children():
        for child_name, child_module in module.named_children():
            if child_name == 'encoder':
                for encoder_layer_name, encoder_module in child_module.named_children():
                    if encoder_layer_name == 'encoder':

                        #Creating Attention Mask
                        attention_mask = 1.0-pre_mask
                        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
                        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

                        encoder_list = list(encoder_module.named_children())[0][1]

                        '''
                        Here we begin to access individual transformer
                        layers. We define some basic variables first.
                        '''

                        #num_attention_heads = config.num_attention_heads
                        #attention_head_size = int(config.hidden_size / config.num_attention_heads)

                        for bert_encoder_layer_name, bert_encoder_module in encoder_list.named_children():
                            # NOTE:
                            # extracting the attention scores of second to last layer
                            if bert_encoder_layer_name == '10':
                                for bert_sublayer_name, bert_sublayer_module in bert_encoder_module.named_children():
                                    if bert_sublayer_name == 'attention':
                                        self_attention_module = list(bert_sublayer_module.children())[0]
                                        self_attention_submodule_list = list(self_attention_module.children())
                                        query, key, value, _ = self_attention_submodule_list

                                        mixed_query_layer = query(output)
                                        mixed_key_layer = key(output)
                                        mixed_value_layer = value(output)

                                        query_layer = transpose_for_scores(mixed_query_layer)
                                        key_layer = transpose_for_scores(mixed_key_layer)
                                        value_layer = transpose_for_scores(mixed_value_layer)

                                        # Take the dot product between "query" and "key" to get the raw attention scores.
                                        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                                        attention_scores = attention_scores / math.sqrt(768) #768 = attention_head_size
                                        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

                                        attention_scores = attention_scores + extended_attention_mask

                                        # Normalize the attention scores to probabilities.
                                        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
                                        for i, bias_indx in enumerate(bias_label_indx):
                                            new_entry = {}
                                            #print("index: {}".format(bias_indx))
                                            #print(sentences[i])
                                            try:
                                                index_pad = sentences[i].index('[PAD]') - 1
                                            except:
                                                index_pad = len(sentences[i]) - 1
                                            new_entry['input_toks'] = sentences[i][:index_pad]
                                            attention_dist = attention_probs[i, :, bias_indx, :].tolist()
                                            new_entry['attention_dist'] = attention_dist[0][:index_pad]
                                            new_entry['full_attention_dist'] = attention_dist[0]
                                            new_entry['probs'] = is_bias_probs[i, :].tolist()
                                            new_entry['labels'] = tok_label_id[i, :].tolist()
                                            results.append(new_entry)
                                        return
                            output = bert_encoder_module(output, extended_attention_mask)
                    output = encoder_module(pre_id)



results = []

for step, batch in enumerate(eval_dataloader):

    if CUDA:
        batch = tuple(x.cuda() for x in batch)
    (
        pre_id, pre_mask, pre_len,
        post_in_id, post_out_id,
        tok_label_id, _,
        rel_ids, pos_ids, categories
    ) = batch

    sentences = [tokenizer.convert_ids_to_tokens(seq) for seq in pre_id.cpu().numpy()]
    bias_label_indx = [labels.index(1) for labels in tok_label_id.tolist()]

    output = None # keeps track of model output
    with torch.no_grad():

        '''
        First finding the probabilities that are returned from the tagger
        '''

        is_bias_probs, _ = joint_model.run_tagger(
            pre_id, pre_mask, rel_ids=rel_ids, pos_ids=pos_ids,
            categories=categories)

        run_attention_extractor()
        print(len(results))
        continue

pickle.dump(results, open("attention_results.pkl", "wb+"))
print("saving out results")
