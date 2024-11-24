from frdfmml.utils import data_loader
from transformers import GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def etl_func(row, **kwargs):

    max_length = kwargs['max_length']
    tokenizer = kwargs['tokenizer']

    tokens_en = tokenizer(
        row['en'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=False
    )

    tokens_fr = tokenizer(
        row['fr'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=False
    )

    tokens_tensor = torch.cat((tokens_en['input_ids'].view(-1), tokens_fr['input_ids'].view(-1)), dim=-1)
    return tokens_tensor


csv_dl = data_loader.csv_data_loader('data/en-fr.csv', 33, etl_func=etl_func, tokenizer=tokenizer, max_length=5)


for tensor in csv_dl:
    print(tensor.size(), tensor.shape)
    print(tensor)