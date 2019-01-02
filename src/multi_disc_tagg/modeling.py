from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel
import torch
import torch.nn as nn


class BertForMultitask(PreTrainedBertModel):

    def __init__(self, config, cls_num_labels=2, tok_num_labels=2):
        super(BertForMultitask, self).__init__(config)
        self.bert = BertModel(config)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)
        
        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)
        
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
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
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        sequence_output = self.tok_dropout(sequence_output)
        tok_logits = self.tok_classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        replace_logits = self.classifier(pooled_output)
        return replace_logits, tok_logits


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
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
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

        return replace_logits, tok_logits




