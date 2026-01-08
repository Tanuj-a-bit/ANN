import torch
from config import IDX_TO_CHAR

def decode_prediction(output):
    """
    Decode CTC output using greedy search.
    """
    # output: (seq_len, batch_size, num_classes)
    # Take argmax across classes
    arg_maxes = torch.argmax(output, dim=2) # (seq_len, batch_size)
    
    decoded_texts = []
    for i in range(arg_maxes.size(1)): # For each item in batch
        indices = arg_maxes[:, i].tolist()
        
        # CTC Decoding: Collapse repeated characters and remove blanks (0)
        decoded = []
        prev = -1
        for idx in indices:
            if idx != 0 and idx != prev:
                decoded.append(IDX_TO_CHAR.get(idx, ''))
            prev = idx
        decoded_texts.append("".join(decoded))
        
    return decoded_texts

def calculate_metrics(preds, targets):
    """
    Calculate Character Error Rate (CER).
    """
    import Levenshtein
    
    total_dist = 0
    total_len = 0
    
    for p, t in zip(preds, targets):
        dist = Levenshtein.distance(p, t)
        total_dist += dist
        total_len += len(t)
        
    cer = total_dist / total_len if total_len > 0 else 0
    return cer
