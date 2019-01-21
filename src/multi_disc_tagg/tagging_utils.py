from tqdm import tqdm
import numpy as np
import torch

NUM_TOK_LABELS = 3


CUDA = (torch.cuda.device_count() > 0)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def run_inference(model, eval_dataloader, tok_criterion, tokenizer):
    out = {
        'input_toks': [],

        'tok_loss': [],
        'tok_logits': [],
        'tok_labels': [],

        'labeling_hits': []
    }

    for step, batch in enumerate(tqdm(eval_dataloader)):
        # if step > 1:
        #     continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)

        ( 
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            tok_label_id, _, 
            replace_id, rel_ids, pos_ids, type_ids
        ) = batch

        with torch.no_grad():
            bias_logits, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
                rel_ids=rel_ids, pos_ids=pos_ids)

            tok_loss = tok_criterion(
                tok_logits.contiguous().view(-1, NUM_TOK_LABELS), 
                tok_label_id.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))
            
        out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in pre_id.cpu().numpy()]

        out['tok_loss'].append(float(tok_loss.cpu().numpy()))
        logits = tok_logits.detach().cpu().numpy()
        labels = tok_label_id.cpu().numpy()
        out['tok_logits'] += logits.tolist()
        out['tok_labels'] += labels.tolist()
        out['labeling_hits'] += tag_hits(logits, labels)

    return out

def classification_accuracy(logits, labels):
    probs = softmax(np.array(logits), axis=1)
    preds = np.argmax(probs, axis=1)
    return metrics.accuracy_score(labels, preds)



def is_ranking_hit(probs, labels, top=1):
    # get rid of padding idx
    [probs, labels] = list(zip(*[(p, l)  for p, l in zip(probs, labels) if l != NUM_TOK_LABELS - 1 ]))
    probs_indices = list(zip(np.array(probs)[:, 1], range(len(labels))))
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    if sum([labels[i] for i in top_indices]) > 0:
        return 1
    else:
        return 0

def tag_hits(logits, tok_labels, top=1):
    probs = softmax(np.array(logits)[:, :, : NUM_TOK_LABELS - 1], axis=2)

    hits = [
        is_ranking_hit(prob_dist, tok_label, top=top) 
        for prob_dist, tok_label in zip(probs, tok_labels)
    ]
    return hits
