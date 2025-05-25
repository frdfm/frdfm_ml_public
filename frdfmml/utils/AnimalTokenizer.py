import torch

class AnimalTokenizer:
    def __init__(self):
        self.vocab = ['<pad>', '<bos>', '<eos>', '<unk>','cat', 'dog', 'pig', 'bat', 'rat', 'fox', 'cow', 'ant', 'elk', 'hen']
        # self.vocab = ['<pad>', '<bos>', '<eos>', '<unk>', 'one', 'two', 'three', 'four', 'five', 'six']
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.bos_token_id = self.token_to_id[self.bos_token]

    def encode(self, text, max_length):
        tokens = text.strip().split()
        # ids = [self.bos_token_id] + [self.token_to_id.get(t, self.token_to_id['<unk>']) for t in tokens] + [self.eos_token_id]
        ids = [self.token_to_id.get(t, self.token_to_id['<unk>']) for t in tokens]
        ids = ids[:max_length]
        ids += [self.pad_token_id] * (max_length - len(ids))
        return ids

    def __call__(self, text, padding='max_length', truncation=True, max_length=16, return_tensors=None, return_attention_mask=False):
        ids = self.encode(text, max_length)
        if return_tensors == "pt":
            return {'input_ids': torch.tensor([ids])}
        return {'input_ids': [ids]}

    def decode(self, ids, skip_special_tokens=True):
        tokens = [self.id_to_token.get(i.item(), '<unk>') for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in ('<pad>', '<bos>', '<eos>')]
        return ' '.join(tokens)

    @property
    def vocab_size(self):
        return len(self.vocab)
