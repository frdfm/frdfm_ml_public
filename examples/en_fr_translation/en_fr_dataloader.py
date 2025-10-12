from random import randint
import torch


def etl_func(row, **kwargs):
    max_length = kwargs['max_length']
    tokenizer = kwargs['tokenizer']

    row_en, row_fr = (row[0], row[1])

    tokens_en = tokenizer(
        row_en,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=False
    )

    tokens_fr = tokenizer(
        row_fr,
        padding='max_length',
        truncation=True,
        max_length=max_length - 1,
        return_tensors="pt",
        return_attention_mask=False
    )

    # tokens_en['input_ids'] = torch.cat([torch.tensor([[tokenizer.eos_token_id]]), tokens_en['input_ids']], dim=1)
    tokens_fr['input_ids'] = torch.cat([torch.tensor([[0]]), tokens_fr['input_ids']], dim=1)

    tokens_tensor = torch.cat((tokens_en['input_ids'].view(-1), tokens_fr['input_ids'].view(-1)), dim=-1)
    return tokens_tensor


# Uncomment for standalone testing
#
# from frdfmml.utils import data_loader
# from transformers import GPT2Tokenizer
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.bos_token = "<|bos|>"
# csv_dl = data_loader.csv_data_loader('../data/en-fr.csv', 1_000_000, etl_func=etl_func, tokenizer=tokenizer, max_length=32, chunk_size=64 * 1024, batch_size=32)
#
# for tensor in csv_dl:
#     print(tensor.size())
#     #print(tensor)