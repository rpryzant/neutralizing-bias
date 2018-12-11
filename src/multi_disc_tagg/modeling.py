from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel

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
