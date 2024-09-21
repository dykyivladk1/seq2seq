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

args = parser.parse_args()



train_df = pd.read_csv(args.data_path, nrows=50)
articles = train_df['article'].astype(str).tolist()
highlights = train_df['highlights'].astype(str).tolist()


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')



def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Training Epochs"):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            src = batch['input_ids'].to(device) 
            tgt = batch['labels'].to(device)  

            tgt_input = tgt[:, :-1]  
            targets = tgt[:, 1:].reshape(-1)  

            outputs = model(src, tgt_input) 
            outputs = outputs.reshape(-1, outputs.size(-1)) 

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')




max_article_len = 512
max_highlight_len = 128

print("Tokenizing articles...")
article_encodings = tokenizer(
    articles,
    max_length=max_article_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)



print("Tokenizing highlights...")
highlight_encodings = tokenizer(
    highlights,
    max_length=max_highlight_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

input_ids = article_encodings['input_ids']
attention_masks = article_encodings['attention_mask']
target_ids = highlight_encodings['input_ids']
target_attention_masks = highlight_encodings['attention_mask']



dataset = CNNDailyMailDataset(input_ids, attention_masks, target_ids, target_attention_masks)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


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



criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = args.epochs

print("Starting training...")

device = decider()

train_model(model, dataloader, criterion, optimizer, num_epochs, device)

