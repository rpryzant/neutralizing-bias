from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertSelfAttention
import torch
import torch.nn as nn
import numpy as np
import ops

import feature_extractors


class BertForMultitask(PreTrainedBertModel):

    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(BertForMultitask, self).__init__(config)
        self.bert = BertModel(config)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)
        
        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)
        
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output, attn_maps = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.cls_dropout(sequence_output)
        pooled_output = self.tok_dropout(pooled_output)
        
        cls_logits = self.cls_classifier(pooled_output)
        tok_logits = self.tok_classifier(sequence_output)

        return cls_logits, tok_logits




class BertForMultitaskWithFeatures(PreTrainedBertModel):

    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(BertForMultitaskWithFeatures, self).__init__(config)
        self.bert = BertModel(config)
        
        self.featurizer = feature_extractors.Featurizer(tok2id) 


        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)
        
        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)
        
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        features = self.featurizer.featurize_batch(
            input_ids.detach().cpu().numpy(), padded_len=input_ids.shape[1])

        sequence_output, pooled_output, attn_maps = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.cls_dropout(sequence_output)
        pooled_output = self.tok_dropout(pooled_output)
        
        cls_logits = self.cls_classifier(pooled_output)
        tok_logits = self.tok_classifier(sequence_output)

        return cls_logits, tok_logits


































class BertForReplacementCLS(PreTrainedBertModel):
    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, replace_num_labels=30522):
        super(BertForReplacementCLS, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, replace_num_labels)

        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tok_indices=None):
        # attn probs are [batch, heads, origin token, distribution]
        sequence_output, pooled_output, layerwise_attn_probs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # attn probs are [layer, batch, heads, origin token, distribution]
        attn_probs = np.array([x.detach().cpu().numpy() for x in layerwise_attn_probs])

        sequence_output = self.tok_dropout(sequence_output)
        tok_logits = self.tok_classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        replace_logits = self.classifier(pooled_output)
        return replace_logits, tok_logits, attn_probs


class BertForReplacementTOK(PreTrainedBertModel):
    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, replace_num_labels=30522):
        super(BertForReplacementTOK, self).__init__(config)
        self.bert = BertModel(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, replace_num_labels)        

        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tok_indices=None):
        sequence_output, pooled_output, layerwise_attn_probs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # attn probs are [layer, batch, heads, origin token, distribution]
        attn_probs = np.array([x.detach().cpu().numpy() for x in layerwise_attn_probs])

        # select all the thingies inside
        pooled_output = []
        for tensor, label_index in zip(sequence_output, tok_indices):
            pooled_output.append(tensor[label_index])

        pooled_output = torch.stack(pooled_output)

        x = self.dense(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        replace_logits = self.classifier(x)

        sequence_output = self.tok_dropout(sequence_output)
        tok_logits = self.tok_classifier(sequence_output)

        return replace_logits, tok_logits, attn_probs



class BertForReplacementTOKAttnClassifier(PreTrainedBertModel):
    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, replace_num_labels=30522,
        attn_type='bilinear', args=None):
        super(BertForReplacementTOKAttnClassifier, self).__init__(config)
        self.bert = BertModel(config)

        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, replace_num_labels)        

        self.del_embedding = nn.Embedding(1, config.hidden_size)
        
        self.attn_type = attn_type
        if attn_type == 'bilinear':
            self.vocab_attn = ops.BilinearAttention(config.hidden_size, 'bahdanau')
        elif attn_type == 'bert':
            self.vocab_attn = ops.BertAttention(config, 
                dropout_scores=args.attn_dropout, compress_heads=args.compress_heads)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tok_indices=None):
        sequence_output, pooled_output, layerwise_attn_probs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # attn probs are [layer, batch, heads, origin token, distribution]
        attn_probs = np.array([x.detach().cpu().numpy() for x in layerwise_attn_probs])

        # select all the thingies inside
        pooled_output = []
        for tensor, label_index in zip(sequence_output, tok_indices):
            pooled_output.append(tensor[label_index])

        pooled_output = torch.stack(pooled_output)

        vocab = torch.cat((self.bert.embeddings.word_embeddings.weight, self.del_embedding.weight), 0)
        batch_size = pooled_output.shape[0]
        vocab = vocab.unsqueeze(0)

        # unroll the batch because duplicating won't fit in memory :(
        replace_logits_tmp = []
        for selected_output in pooled_output:
            replace_logits_tmp.append(
                self.vocab_attn(query=selected_output.unsqueeze(0), keys=vocab)
            )

        replace_logits = torch.stack(replace_logits_tmp)
        # x = self.dense(pooled_output)
        # x = self.activation(x)
        # x = self.dropout(x)
        # replace_logits = self.classifier(x)

        sequence_output = self.tok_dropout(sequence_output)
        tok_logits = self.tok_classifier(sequence_output)

        return replace_logits, tok_logits, attn_probs



