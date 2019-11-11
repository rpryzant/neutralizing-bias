from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertSelfAttention
import pytorch_pretrained_bert.modeling as modeling
import torch
import torch.nn as nn
import numpy as np
import copy

import sys; sys.path.append('.')
import sys; sys.path.append('tagging/')   # so that the joint model can see this filter
import features
from shared.args import ARGS
from shared.constants import CUDA
import seq2seq.model as seq2seq_model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def identity(x):
    return x

class BertForMultitask(PreTrainedBertModel):

    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(BertForMultitask, self).__init__(config)
        self.bert = BertModel(config)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)
        
        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)
        
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        labels=None, rel_ids=None, pos_ids=None, categories=None, pre_len=None):
        global ARGS
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        cls_logits = self.cls_classifier(pooled_output)
        cls_logits = self.cls_dropout(cls_logits)

        # NOTE -- dropout is after proj, which is non-standard
        tok_logits = self.tok_classifier(sequence_output)
        tok_logits = self.tok_dropout(tok_logits)

        return cls_logits, tok_logits




class ConcatCombine(nn.Module):
    def __init__(self, hidden_size, feature_size, out_size, layers,
            dropout_prob, small=False, pre_enrich=False, activation=False,
            include_categories=False, category_emb=False,
            add_category_emb=False):
        super(ConcatCombine, self).__init__()

        self.include_categories = include_categories
        self.add_category_emb = add_category_emb
        if include_categories:
            if category_emb and not add_category_emb:
                feature_size *= 2
            elif not category_emb:
                feature_size += 43

        if layers == 1:
            self.out = nn.Sequential(
                nn.Linear(hidden_size + feature_size, out_size),
                nn.Dropout(dropout_prob))
        elif layers == 2:
            waist_size = min(hidden_size, feature_size) if small else max(hidden_size, feature_size)
            if activation:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size + feature_size, waist_size),
                    nn.Dropout(dropout_prob),
                    nn.ReLU(),
                    nn.Linear(waist_size, out_size),
                    nn.Dropout(dropout_prob))
            else:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size + feature_size, waist_size),
                    nn.Dropout(dropout_prob),
                    nn.Linear(waist_size, out_size),
                    nn.Dropout(dropout_prob))
        if pre_enrich:
            if activation:
                self.enricher = nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    nn.ReLU())
            else:
                self.enricher = nn.Linear(feature_size, feature_size)
        else:
            self.enricher = None
        # manually set cuda because module doesn't see these combiners for bottom 
        if CUDA:
            self.out = self.out.cuda()
            if self.enricher: 
                self.enricher = self.enricher.cuda()
                
    def forward(self, hidden, features, categories=None):
        if self.include_categories:
            categories = categories.unsqueeze(1)
            categories = categories.repeat(1, features.shape[1], 1)
            if self.add_category_emb:
                features = features + categories
            else:
                features = torch.cat((features, categories), -1)

        if self.enricher is not None:
            features = self.enricher(features)

        return self.out(torch.cat((hidden, features), dim=-1))


class AddCombine(nn.Module):
    def __init__(self, hidden_dim, feat_dim, layers, dropout_prob, small=False,
            out_dim=-1, pre_enrich=False, include_categories=False,
            category_emb=False, add_category_emb=False):
        super(AddCombine, self).__init__()

        self.include_categories = include_categories
        if include_categories:
            feat_dim += 43

        if layers == 1:
            self.expand = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.Dropout(dropout_prob))
        else:
            waist_size = min(feat_dim, hidden_dim) if small else max(feat_dim, hidden_dim)
            self.expand = nn.Sequential(
                nn.Linear(feat_dim, waist_size),
                nn.Dropout(dropout_prob),
                nn.Linear(waist_size, hidden_dim),
                nn.Dropout(dropout_prob))
        
        if out_dim > 0:
            self.out = nn.Linear(hidden_dim, out_dim)
        else:
            self.out = None

        if pre_enrich:
            self.enricher = nn.Linear(feature_size, feature_size)        
        else:
            self.enricher = None

        # manually set cuda because module doesn't see these combiners for bottom         
        if CUDA:
            self.expand = self.expand.cuda()
            if out_dim > 0:
                self.out = self.out.cuda()
            if self.enricher is not None:
                self.enricher = self.enricher.cuda()

    def forward(self, hidden, feat, categories=None):
        if self.include_categories:
            categories = categories.unsqueeze(1)
            categories = categories.repeat(1, features.shape[1], 1)
            if self.add_category_emb:
                features = features + categories
            else:
                features = torch.cat((features, categories), -1)

        if self.enricher is not None:
            feat = self.enricher(feat)
    
        combined = self.expand(feat) + hidden
    
        if self.out is not None:
            return self.out(combined)

        return combined


class BertForMultitaskWithFeaturesOnTop(PreTrainedBertModel):
    """ stick the features on top of the model """
    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(BertForMultitaskWithFeaturesOnTop, self).__init__(config)
        global ARGS
        
        self.bert = BertModel(config)
        
        self.featurizer = features.Featurizer(
            tok2id, lexicon_feature_bits=ARGS.lexicon_feature_bits) 
        # TODO -- don't hardcode this...
        nfeats = 90 if ARGS.lexicon_feature_bits == 1 else 118

        if ARGS.extra_features_method == 'concat':
            self.tok_classifier = ConcatCombine(
                config.hidden_size, nfeats, tok_num_labels, 
                ARGS.combiner_layers, config.hidden_dropout_prob,
                ARGS.small_waist, pre_enrich=ARGS.pre_enrich,
                activation=ARGS.activation_hidden,
                include_categories=ARGS.concat_categories,
                category_emb=ARGS.category_emb,
                add_category_emb=ARGS.add_category_emb)
        else:
            self.tok_classifier = AddCombine(
                config.hidden_size, nfeats, ARGS.combiner_layers,
                config.hidden_dropout_prob, ARGS.small_waist,
                out_dim=tok_num_labels, pre_enrich=ARGS.pre_enrich,
                include_categories=ARGS.concat_categories,
                category_emb=ARGS.category_emb,
                add_category_emb=ARGS.add_category_emb)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)

        self.category_emb = ARGS.category_emb
        if ARGS.category_emb:
            self.category_embeddings = nn.Embedding(43, nfeats)

        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        labels=None, rel_ids=None, pos_ids=None, categories=None, pre_len=None):
        global ARGS
        global CUDA

        features = self.featurizer.featurize_batch(
            input_ids.detach().cpu().numpy(), 
            rel_ids.detach().cpu().numpy(), 
            pos_ids.detach().cpu().numpy(), 
            padded_len=input_ids.shape[1])
        features = torch.tensor(features, dtype=torch.float)
        if CUDA:
            features = features.cuda()

        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.cls_dropout(pooled_output)
        cls_logits = self.cls_classifier(pooled_output)

        if ARGS.category_emb:
            categories = self.category_embeddings(
                categories.max(-1)[1].type(
                    'torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

        tok_logits = self.tok_classifier(sequence_output, features, categories)

        return cls_logits, tok_logits


class TaggerFromDebiaser(nn.Module):
    def __init__(self, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(TaggerFromDebiaser, self).__init__()

        global ARGS
        global CUDA

        if ARGS.pointer_generator:
            self.debias_model = seq2seq_model.PointerSeq2Seq(
                vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
                emb_dim=768, dropout=0.2, tok2id=tok2id)
        else:
            self.debias_model = seq2seq_model.Seq2Seq(
                vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
                emb_dim=768, dropout=0.2, tok2id=tok2id)

        assert ARGS.debias_checkpoint
        print('LOADING DEBIASER FROM ' + ARGS.debias_checkpoint)
        self.debias_model.load_state_dict(torch.load(ARGS.debias_checkpoint))
        print('...DONE')

        self.cls_classifier = nn.Sequential(
            nn.Linear(ARGS.hidden_size, ARGS.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(ARGS.hidden_size, cls_num_labels),
            nn.Dropout(0.1))

        self.tok_classifier = nn.Sequential(
            nn.Linear(ARGS.hidden_size, ARGS.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(ARGS.hidden_size, tok_num_labels),
            nn.Dropout(0.1))


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
        labels=None, rel_ids=None, pos_ids=None, categories=None, pre_len=None):

        pre_mask = 1.0-attention_mask

        # src_outputs is [batch_size, sequence_length, hidden_size].
        src_outputs, h_t, _ = self.debias_model.run_encoder(
            input_ids, pre_len, pre_mask)

        cls_logits = self.cls_classifier(h_t)
        tok_logits = self.tok_classifier(src_outputs)

        return cls_logits, tok_logits
