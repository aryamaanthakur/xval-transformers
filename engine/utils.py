import os
import random

import torch
import numpy as np

PAD_IDX = 0

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (torch.zeros((src.shape[0], src_seq_len), device=device)).type(torch.bool)
    tgt_padding_mask = (tgt == PAD_IDX)
    tgt_mask = tgt_mask
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def sequence_accuracy(y_pred, y_true):

    count = 0
    total = len(y_pred)
    for (predicted_tokens, original_tokens) in zip(y_pred, y_true):
        original_tokens = original_tokens.tolist()
        predicted_tokens = predicted_tokens.tolist()
        if original_tokens == predicted_tokens:
            count = count+1

    return count/total
