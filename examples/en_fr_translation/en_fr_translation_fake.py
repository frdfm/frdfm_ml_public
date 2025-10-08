from frdfm_ml.utils import data_loader
from transformers import GPT2Tokenizer, AutoTokenizer
from en_fr_dataloader import etl_func
import torch
import torch.nn as nn
import torch.optim as optim
from frdfm_ml.utils.gen_translation_utils import generate_translation
from frdfm_ml.utils.AnimalTokenizer import AnimalTokenizer

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer = AnimalTokenizer()
tokenizer.pad_token = tokenizer.eos_token

# print(tokenizer.bos_token, tokenizer.bos_token_id)
# print(tokenizer.eos_token, tokenizer.eos_token_id)
# print(tokenizer.pad_token, tokenizer.pad_token_id)

vocab_size = tokenizer.vocab_size
seq_size = 5#16  # Maximum size of the sequence
batch_size = 1 #256 // 4
chunk_size = 1000  # How many rows to cache
num_head = 6
num_layers = 6
emb_size = num_head * 64 # 256 * 3 * 2
lr = 0.0001  # Initial lr # a good one 0.00001
lr_decay_rate = 0.99997698 # a good one .99999
num_epoch = 1000000
ds_size = 1000_000  # Number of healthy rows in dataset
pr_inter = 100  # Print interval
eval_inter = 1000  # Eval interval


import torch.nn.functional as F
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, num_head, num_layers, seq_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(0.1)
        self.positional_encoding_src = nn.Parameter(torch.zeros(seq_size, emb_size), requires_grad=True)
        self.positional_encoding_tgt = nn.Parameter(torch.zeros(seq_size, emb_size), requires_grad=True)
        self.transformer = nn.Transformer(emb_size, num_head, num_layers)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.dropout(self.embedding(src) + self.positional_encoding_src)
        tgt_emb = self.dropout(self.embedding(tgt) + self.positional_encoding_tgt)
        out = self.transformer(src_emb, tgt_emb)
        return self.fc(out)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size=vocab_size, emb_size=emb_size, num_head=num_head, num_layers=num_layers, seq_size=seq_size).to(device)
model.device = device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

for epoch in range(num_epoch):

    dl = data_loader.csv_data_loader('../../data/data.csv', ds_size, etl_func=etl_func, tokenizer=tokenizer, max_length=seq_size, chunk_size=chunk_size, batch_size=batch_size)
    for i, batch in enumerate(dl):

        # print(tokenizer.decode(batch[0, :], skip_special_tokens=True))

        model.train()
        optimizer.zero_grad()

        src, tgt = batch[:, :batch.size(-1) // 2].to(device), batch[:, batch.size(-1) // 2:].to(device)
        tgt_input = tgt[:, 1:]

        output = model(src, tgt)
        output_for_val = output[:, :-1, :]

        loss = criterion(output_for_val.contiguous().view(-1, vocab_size), tgt_input.contiguous().view(-1))
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

        del src, tgt, tgt_input, output, output_for_val, loss

        for param_group in optimizer.param_groups:
            if i % pr_inter == 0:
                print(f"{i:10d} :: {i * batch_size:10d} :: {param_group['lr']:12.10f} :: {loss_value:9.6f}")

            param_group['lr'] = param_group['lr'] * lr_decay_rate

        if i % eval_inter == 0:
            model.eval()
            with torch.no_grad():
                txts = [
                    "dog cat",
                    # "one two three four",
                    # "one two five six",
                    # "one two one one",
                    # "two six six six",
                    # "five four three",
                    # "five six one one",
                    # "one one one one",
                    # "two two two two",
                    "elk dog cat",
                    "pig pig pig ant"
                ]
                for jj in range(1, 2):
                    for txt in txts:
                        print(txt, "-->", generate_translation(model, tokenizer, txt, seq_size, jj, jj))
