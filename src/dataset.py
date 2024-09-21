import torch
from torch.utils.data import Dataset




class CNNDailyMailDataset(Dataset):
    def __init__(self, input_ids, attention_masks, target_ids, target_attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.target_ids = target_ids
        self.target_attention_masks = target_attention_masks

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.target_ids[idx],
            'labels_attention_mask': self.target_attention_masks[idx]
        }


