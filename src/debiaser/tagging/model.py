from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertSelfAttention, BertEmbeddings, BertEncoder, BertPooler
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



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def identity(x):
    return x

class BertModelWithFeatureSignal(BertModel):
    """Adapted from pytorch_pretrained_bert.modeling.BertModel.
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, feature_size):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.feature_embeddings = nn.Embedding(feature_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, feature_ids=None, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        if feature_ids is not None:
            feature_signal = self.feature_embeddings(feature_ids)
            # Match the sequence length dimension to the input.
            feature_signal = feature_signal.unsqueeze(1).expand_as(embedding_output)
            embedding_output += feature_signal
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


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
        labels=None, rel_ids=None, pos_ids=None, categories=None):
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
        
        if ARGS.category_signal:
            self.bert = BertModelWithFeatureSignal(config, ARGS.num_categories)
        else:
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
          size = ARGS.category_emb_size if ARGS.category_emb_size else nfeats
            self.category_embeddings = nn.Embedding(ARGS.num_categories, size)

        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        labels=None, rel_ids=None, pos_ids=None, categories=None):
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

        if ARGS.category_signal:
            category_ids = categories.max(-1)[1].type(
                'torch.cuda.LongTensor' if CUDA else 'torch.LongTensor')
            sequence_output, pooled_output = self.bert(
                input_ids, category_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        else:
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


class BertForMultitaskWithFeaturesOnBottom(PreTrainedBertModel):
    """ stick the features on top of the model """
    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None, args=None):
        super(BertForMultitaskWithFeaturesOnBottom, self).__init__(config)
        global ARGS
        
        self.featurizer = features.Featurizer(
            tok2id, lexicon_feature_bits=ARGS.lexicon_feature_bits) 
        # TODO -- don't hardcode this...
        nfeats = 90 if ARGS.lexicon_feature_bits == 1 else 118

        if ARGS.extra_features_method == 'concat':
            if ARGS.share_combiners:
                self.combiners = {
                    i: ConcatCombine(
                        config.hidden_size, nfeats, config.hidden_size, 
                        ARGS.combiner_layers, config.hidden_dropout_prob,
                        ARGS.small_waist, pre_enrich=ARGS.pre_enrich,
                        activation=ARGS.activation_hidden)
                    for i in range(1, 7)
                }
            else:
                combiner = ConcatCombine(
                    config.hidden_size, nfeats, config.hidden_size, 
                    ARGS.combiner_layers, config.hidden_dropout_prob,
                    ARGS.small_waist, pre_enrich=ARGS.pre_enrich,
                    activation=ARGS.activation_hidden)
                self.combiners = { i: combiner for i in range(1, 7) }
        else:
            if ARGS.share_combiners:
                self.combiners = {
                    i: AddCombine(
                nfeats, config.hidden_size, ARGS.combiner_layers,
                config.hidden_dropout_prob, ARGS.small_waist, 
                pre_enrich=ARGS.pre_enrich)
                    for i in range(1, 7)
                }
            else:
                combiner = AddCombine(
                    nfeats, config.hidden_size, ARGS.combiner_layers,
                    config.hidden_dropout_prob, ARGS.small_waist)
                self.combiners = { i: combiner for i in range(1, 7) }

        self.bert = BertModelBottomFeatures(config, self.combiners)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)

        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)

        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        labels=None, rel_ids=None, pos_ids=None, categories=None):

        features = self.featurizer.featurize_batch(
            input_ids.detach().cpu().numpy(),
            rel_ids.detach().cpu().numpy(),
            pos_ids.detach().cpu().numpy(),
            padded_len=input_ids.shape[1])
        features = torch.tensor(features, dtype=torch.float)
        if CUDA:
            features = features.cuda()
        
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False,
            features=features)

        sequence_output = self.cls_dropout(sequence_output)
        cls_logits = self.cls_classifier(pooled_output)

        # NOTE -- dropout is after proj, which is non-standard
        #      -- switch back if nessicary!
        tok_logits = self.tok_classifier(sequence_output)
        tok_logits = self.tok_dropout(tok_logits)

        return cls_logits, tok_logits











class BertModelBottomFeatures(BertModel):
    def __init__(self, config, combiners):
        super(BertModelBottomFeatures, self).__init__(config)
        self.embeddings = modeling.BertEmbeddings(config)
        self.encoder = BertEncoderF(config, combiners)
        self.pooler = modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, features=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, layerwise_attn_probs = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      features=features)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, layerwise_attn_probs




class BertSelfOutputF(nn.Module):
    def __init__(self, config, combiners):
        super(BertSelfOutputF, self).__init__()
        self.combiners = combiners
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = modeling.BertLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, features=None):
        global ARGS
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        ### COMBINE1
        if features is not None and ARGS.combine1:
            hidden_states = self.combiners[1](hidden_states, features)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttentionF(nn.Module):
    def __init__(self, config, combiners):
        super(BertAttentionF, self).__init__()
        self.combiners = combiners
        self.self = modeling.BertSelfAttention(config)
        self.output = BertSelfOutputF(config, combiners)

    def forward(self, input_tensor, attention_mask, features=None):
        global ARGS
        
        self_output, attn_probs = self.self(input_tensor, attention_mask)

        ### COMBINE2
        if features is not None and ARGS.combine2:
            self_output = self.combiners[2](self_output, features)

        attention_output = self.output(self_output, input_tensor, features=features)
        return attention_output, attn_probs


class BertOutputF(nn.Module):
    def __init__(self, config, combiners):
        super(BertOutputF, self).__init__()
        self.combiners = combiners
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = modeling.BertLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, features=None):
        global ARGS
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        ### COMBINE3
        if features is not None and ARGS.combine3:
            hidden_states = self.combiners[3](hidden_states, features)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayerF(nn.Module):
    def __init__(self, config, combiners):
        super(BertLayerF, self).__init__()
        self.combiners = combiners
        self.attention = BertAttentionF(config, combiners)
        self.intermediate = modeling.BertIntermediate(config)
        self.output = BertOutputF(config, combiners)

    def forward(self, hidden_states, attention_mask, features=None):
        global ARGS
        ### COMBINE4        
        if features is not None and ARGS.combine4:
            hidden_states = self.combiners[4](hidden_states, features)

        attention_output, attn_probs = self.attention(hidden_states, attention_mask, features=features)
        
        ### COMBINE5
        if features is not None and ARGS.combine5:
            hidden_states = self.combiners[5](hidden_states, features)

        intermediate_output = self.intermediate(attention_output)
        
        ### COMBINE6
        if features is not None and ARGS.combine4:
            hidden_states = self.combiners[6](hidden_states, features)

        layer_output = self.output(intermediate_output, attention_output, features=features)
        return layer_output, attn_probs


class BertEncoderF(nn.Module):
    def __init__(self, config, combiners):
        super(BertEncoderF, self).__init__()
        layer = BertLayerF(config, combiners)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, features=None):
        all_encoder_layers = []
        all_layer_attns = []
        for i, layer_module in enumerate(self.layer):
            if i == len(self.layer) - 1:
                hidden_states, attn_probs = layer_module(hidden_states, attention_mask, features=features)
            else:
                hidden_states, attn_probs = layer_module(hidden_states, attention_mask)

            all_layer_attns.append(attn_probs)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_layer_attns




