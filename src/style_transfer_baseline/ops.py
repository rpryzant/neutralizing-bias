"""
msc neural stuff
"""
import torch
import torch.nn as nn

from cuda import CUDA

class FcTube(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, dropout):
        super(FcTube, self).__init__()
        if nlayers == 1:
            hidden_dim = output_dim

        self.layers = [nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(), 
            nn.Dropout(p=dropout)
        )]          
        for _ in range(nlayers - 2):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    # nn.ReLU(), 
                    nn.Dropout(p=dropout)
            ))
        if nlayers > 1:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, output_dim),
                    # nn.ReLU(), 
                    nn.Dropout(p=dropout)
            ))
        # TODO -- get this into model.parameters so that we can just do model.cuda()
        if CUDA:
            for i in range(len(self.layers)):
                self.layers[i] = self.layers[i].cuda()


    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X


class FFNN(nn.Module):
    # TODO -- play with activation functions?
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, dropout):
        super(FFNN, self).__init__()
        self.tube = FcTube(input_dim, hidden_dim, output_dim, nlayers, dropout)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X, Y):
        X = self.tube(X)

        loss = self.criterion(X, Y)
        return X, loss


class FeedForwardAttention(nn.Module):
    """ simplest self-attention, run a ffnn over all the hidden states
        to determine scores, then sum up according to scores
    """
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super(FeedForwardAttention, self).__init__()
        self.scorer = FcTube(input_dim, hidden_dim, 1, layers, dropout)
        self.softmax = nn.Softmax()
        self.out_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, query, keys, mask=None, values=None):
        scores = self.scorer(keys).squeeze(2)
        scores = scores.masked_fill(mask, -float('inf'))
        probs = self.softmax(scores)
        probs = probs.unsqueeze(1)
        weighted_context = torch.bmm(probs, keys).squeeze(1)

        return weighted_context, None, probs


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
            
        attn_probs = self.softmax(attn_scores)
        # [Batch, 1, source length]
        attn_probs_transposed = attn_probs.unsqueeze(1)
        # [Batch, hidden]
        weighted_context = torch.bmm(attn_probs_transposed, values).squeeze(1)

        context_query_mixed = torch.cat((weighted_context, query), 1)
        context_query_mixed = self.tanh(self.out_projection(context_query_mixed))

        return weighted_context, context_query_mixed, attn_probs


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


# class FeedForwardAttn