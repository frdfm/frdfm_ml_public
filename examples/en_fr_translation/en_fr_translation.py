import math
import random

from frdfm_ml.utils import data_loader
from transformers import GPT2Tokenizer, AutoTokenizer
from en_fr_dataloader import etl_func
import torch
import torch.nn as nn
import torch.optim as optim
from frdfm_ml.utils.gen_translation_utils import generate_translation
from frdfm_ml.utils.AnimalTokenizer import AnimalTokenizer

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
# tokenizer = AnimalTokenizer()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)



# print(tokenizer.bos_token, tokenizer.bos_token_id)
# print(tokenizer.eos_token, tokenizer.eos_token_id)
# print(tokenizer.pad_token, tokenizer.pad_token_id)

vocab_size = tokenizer.vocab_size
seq_size = 100#7#16  # Maximum size of the sequence
batch_size = 32 #256 // 4
chunk_size = 256 * 100  # How many rows to cache
num_head = 8#12
num_layers = 8#12
emb_size = 512#num_head * 64 # 256 * 3 * 2
lr = 0.00001  # Initial lr # a good one 0.00001
lr_decay_rate = 0.99997698 # a good one .99999
LEARNING_RATE = 0.001
power01 = 1
num_epoch = 100
ds_size = 1_600_000  # Number of healthy rows in dataset
pr_inter = 1  # Print interval
eval_inter = 100  # Eval interval

new_lr = 1/(2_000_000/batch_size)
lr_decay = .1 ** (1/(1*(2_000_000/batch_size)))


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, num_head, num_layers, seq_size, dropout=0.95):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding_src = nn.Parameter(torch.zeros(seq_size, emb_size), requires_grad=True)
        self.positional_encoding_tgt = nn.Parameter(torch.zeros(seq_size, emb_size), requires_grad=True)

        # Add dropout inside transformer
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=num_head,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # External dropout for embeddings
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.embedding(src) + self.positional_encoding_src.unsqueeze(0)
        tgt_emb = self.embedding(tgt) + self.positional_encoding_tgt.unsqueeze(0)

        # Apply dropout on embeddings
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(src.device)

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        #out = self.dropout(out)  # Optional extra dropout on output
        return self.fc(out)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size=vocab_size, emb_size=emb_size, num_head=num_head, num_layers=num_layers, seq_size=seq_size).to(device)
model.device = device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

# Create iterators for eval dataset
dl_eval = data_loader.csv_data_loader_from_two_files(
    '../../data/europarl-v7.fr-en_eval.en',
    '../../data/europarl-v7.fr-en_eval.fr',
    ds_size, etl_func=etl_func, tokenizer=tokenizer, max_length=seq_size, chunk_size=chunk_size, batch_size=batch_size
)
eval_iter = iter(dl_eval)

for epoch in range(num_epoch):

    dl_train = data_loader.csv_data_loader_from_two_files(
        '../../data/europarl-v7.fr-en_train.en',
        '../../data/europarl-v7.fr-en_train.fr',
        ds_size, etl_func=etl_func, tokenizer=tokenizer, max_length=seq_size, chunk_size=chunk_size, batch_size=batch_size
    )

    for i, batch in enumerate(dl_train):

        # --- Training ---
        model.train()
        optimizer.zero_grad()

        src, tgt = batch[:, :batch.size(-1) // 2].to(device), batch[:, batch.size(-1) // 2:].to(device)
        tgt_input = tgt

        output = model(src, tgt, src_padding_mask=(src == tokenizer.pad_token_id),
                       tgt_padding_mask=(tgt == tokenizer.pad_token_id))

        output_for_val = output

        loss = criterion(output_for_val.contiguous().view(-1, vocab_size),
                         tgt_input.contiguous().view(-1))
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

        del src, tgt, tgt_input, output, output_for_val, loss




        model.eval()
        with torch.no_grad():
            try:
                eval_batch = next(eval_iter)
            except:

                dl_eval = data_loader.csv_data_loader_from_two_files(
                    '../../data/europarl-v7.fr-en_eval.en',
                    '../../data/europarl-v7.fr-en_eval.fr',
                    ds_size, etl_func=etl_func, tokenizer=tokenizer, max_length=seq_size, chunk_size=chunk_size, batch_size=batch_size
                )
                eval_iter = iter(dl_eval)
                eval_batch = next(eval_iter)

            src_eval, tgt_eval = eval_batch[:, :eval_batch.size(-1) // 2].to(device), \
                                 eval_batch[:, eval_batch.size(-1) // 2:].to(device)
            tgt_input_eval = tgt_eval[:, 1:]

            output_eval = model(src_eval, tgt_eval)
            output_for_val_eval = output_eval[:, :-1, :]

            eval_loss = criterion(output_for_val_eval.contiguous().view(-1, vocab_size),
                                  tgt_input_eval.contiguous().view(-1))
            eval_loss_value = eval_loss.item()

            del src_eval, tgt_eval, tgt_input_eval, output_eval, output_for_val_eval, eval_loss

        for param_group in optimizer.param_groups:
            if i % pr_inter == 0:
                print(f"{epoch:10d} :: {i * batch_size:10d} :: {param_group['lr']:22.20f} :: Train Loss = {loss_value:9.6f} :: Eval Loss = {eval_loss_value:9.6f} :: {generate_translation(model, tokenizer, 'It has been one of the major priorities of the Irish presidency to secure more coordinated and effective action at European Union level against drug trafficking and drug abuse.', seq_size, 1, 1)}")
            #new_lr = (math.exp(eval_loss_value) / 50_000.) ** power01 * LEARNING_RATE
            #new_lr = 1/10**(random.randint(4+epoch,7+epoch))
            new_lr *= lr_decay
            param_group['lr'] = new_lr#param_group['lr'] * lr_decay_rate

