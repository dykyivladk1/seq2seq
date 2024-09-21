import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import TransformerSeq2Seq
from dataset import CNNDailyMailDataset



from polip import decider


import pandas as pd
from tqdm import tqdm

from transformers import BartTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = 'cnn_dailymail/train.csv')
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--max_article_len', type = int, default = 512)
parser.add_argument('--max_highlight_len', type = int, default = 128)
parser.add_argument('--embed_size', type=int, default=512)
parser.add_argument('--num_encoder_layers', type=int, default=6)
parser.add_argument('--num_decoder_layers', type=int, default=6)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--max_len', type=int, default = 512)
parser.add_argument('--ff_hidden_mult', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--epochs', type = int, default = 30)
parser.add_argument('--sample_text', type = int, default = 30)

args = parser.parse_args()


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = tokenizer.vocab_size



model = TransformerSeq2Seq(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    embed_size=args.embed_size,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    n_head=args.n_head,
    max_len=args.max_len,
    ff_hidden_mult=args.ff_hidden_mult,
    dropout=args.dropout,
    tokenizer_pad_token_id = tokenizer.pad_token_id
)



def generate_summary(model, tokenizer, article, device, max_summary_len=128, top_k=50):
    model.eval()
    with torch.no_grad():
        src_encoding = tokenizer(article, max_length=args.max_len, padding='max_length', truncation=True, return_tensors='pt')
        src = src_encoding['input_ids'].to(device)
        src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
        tgt_input = torch.tensor([[tokenizer.bos_token_id]], device=device)

        for _ in range(max_summary_len):
            output = model(src, tgt_input)
            next_token_logits = output[:, -1, :]
            top_k = max(top_k, 1)
            topk_probs, topk_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), top_k)
            topk_probs = topk_probs.squeeze(0)
            topk_indices = topk_indices.squeeze(0)
            topk_probs = topk_probs / topk_probs.sum()
            next_token = torch.multinomial(topk_probs, num_samples=1).unsqueeze(0)
            next_token = topk_indices[next_token]
            tgt_input = torch.cat([tgt_input, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        summary = tokenizer.decode(tgt_input.squeeze(), skip_special_tokens=True)
        return summary



print(generate_summary(model, tokenizer, article = args.sample_text, device = torch.device('cpu')))