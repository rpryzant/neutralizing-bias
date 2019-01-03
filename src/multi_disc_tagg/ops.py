import torch
import torch.nn as nn
import math

class BilinearAttention(nn.Module):
    """ bilinear attention layer: score(H_j, q) = H_j^T W_a q
                (where W_a = self.in_projection)
    """
    def __init__(self, hidden, score_fn='dot'):
        super(BilinearAttention, self).__init__()
        self.query_in_projection = nn.Linear(hidden, hidden)
        self.key_in_projection = nn.Linear(hidden, hidden)
        self.softmax = nn.Softmax()
        self.out_projection = nn.Linear(hidden * 2, hidden)
        self.tanh = nn.Tanh()
        self.score_fn = self.dot

        if score_fn == 'bahdanau':
            self.v_att = nn.Linear(hidden, 1, bias=False)
            self.score_tanh = nn.Tanh()
            self.score_fn = self.bahdanau

    def forward(self, query, keys, mask=None, values=None):
        """
            query: [batch, hidden]
            keys: [batch, len, hidden]
            values: [batch, len, hidden] (optional, if none will = keys)
            mask: [batch, len] mask key-scores

            compare query to keys, use the scores to do weighted sum of values
            if no value is specified, then values = keys
        """
        att_keys = self.key_in_projection(keys)

        if values is None:
            values = att_keys

        # [Batch, Hidden, 1]
        att_query = self.query_in_projection(query)

        # [Batch, Source length]
        attn_scores = self.score_fn(att_keys, att_query)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -float('inf'))
            
        return attn_scores

    def dot(self, keys, query):
        """
        keys: [B, T, H]
        query: [B, H]
        """
        return torch.bmm(keys, query.unsqueeze(2)).squeeze(2)


    def bahdanau(self, keys, query):
        """
        keys: [B, T, H]
        query: [B, H]
        """
        return self.v_att(self.score_tanh(keys + query.unsqueeze(1))).squeeze(2)



class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, keys, attention_mask=None):
        values = keys

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(keys)
        mixed_value_layer = self.value(values)

        # there's just one query vector and transpose_for_scores expects a sequence of them
        query_layer = self.transpose_for_scores(mixed_query_layer.unsqueeze(1))
        print('here', mixed_key_layer.shape)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        return attention_scores
