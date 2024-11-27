from frdfmml.utils import data_loader
from transformers import GPT2Tokenizer, AutoTokenizer
from en_fr_dataloader import etl_func
import torch
import torch.nn as nn
import torch.optim as optim
from frdfmml.utils.gen_translation_utils import generate_translation

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer.pad_token = tokenizer.eos_token

# print(tokenizer.bos_token, tokenizer.bos_token_id)
# print(tokenizer.eos_token, tokenizer.eos_token_id)
# print(tokenizer.pad_token, tokenizer.pad_token_id)

vocab_size = tokenizer.vocab_size
seq_size = 16  # Maximum size of the sequence
batch_size = 256
chunk_size = 64 * 1024  # How many rows to cache
emb_size = 256 * 3
num_head = 12
num_layers = 12
lr = 0.001  # Initial lr
lr_decay_rate = .9999
num_epoch = 10
ds_size = 1_000_000  # Number of healthy rows in dataset
pr_inter = 10  # Print interval
eval_inter = 100  # Eval interval


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, num_head, num_layers, seq_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = nn.Parameter(torch.zeros(seq_size, emb_size), requires_grad=True)
        self.transformer = nn.Transformer(emb_size, num_head, num_layers)
        self.fc = nn.Linear(emb_size, vocab_size)

        # Optional
        # self.fc.weight = self.embedding.weight

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.positional_encoding
        tgt_emb = self.embedding(tgt) + self.positional_encoding
        out = self.transformer(src_emb, tgt_emb)
        return self.fc(out)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size=vocab_size, emb_size=emb_size, num_head=num_head, num_layers=num_layers, seq_size=seq_size).to(device)
model.device = device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):

    dl = data_loader.csv_data_loader('../data/en-fr.csv', ds_size, etl_func=etl_func, tokenizer=tokenizer, max_length=seq_size, chunk_size=chunk_size, batch_size=batch_size)
    for i, batch in enumerate(dl):

        model.train()
        optimizer.zero_grad()

        src, tgt = batch[:, :batch.size(-1) // 2].to(device), batch[:, batch.size(-1) // 2:].to(device)
        tgt_input = tgt[:, 1:]

        output = model(src, tgt)
        output_for_val = output[:, :-1, :]

        loss = criterion(output_for_val.contiguous().view(-1, vocab_size), tgt_input.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            if i % pr_inter == 0:
                print(f"{i:10d} :: {i * batch_size:10d} :: {param_group['lr']:10.8f} :: {loss.item():9.6f}")

            param_group['lr'] = param_group['lr'] * lr_decay_rate

        if i % eval_inter == 0:
            model.eval()
            with torch.no_grad():
                for j in range(1, 4):
                    print("Hi", "-->", generate_translation(model, tokenizer, "Hi", seq_size, j, j))
                    print("Hi, I am a student.", "-->", generate_translation(model, tokenizer, "Hi, I am a student.", seq_size, j, j))
                    print("How are you?", "-->", generate_translation(model, tokenizer, "How are you?", seq_size, j, j))
