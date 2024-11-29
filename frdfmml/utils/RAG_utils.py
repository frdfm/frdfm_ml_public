import sqlite3
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_embedding(model, tokenizer, seq_size, text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=seq_size)
    outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def store_embeddings(model, tokenizer, seq_size, chunks):
    conn = sqlite3.connect("local_data.db")
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (chunk TEXT, embedding BLOB)")
    for chunk in chunks:
        if not chunk is None and not chunk == '':
            embedding = get_embedding(model, tokenizer, seq_size, chunk).tobytes()
            print(f':: {chunk}')
            conn.execute("INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)", (chunk, embedding))
    conn.commit()
    conn.close()


def retrieve_similar(model, tokenizer, seq_size, query, top_n=3):
    query_embed = get_embedding(model, tokenizer, seq_size, query)
    conn = sqlite3.connect("local_data.db")
    rows = conn.execute("SELECT chunk, embedding FROM embeddings").fetchall()
    similarities = []
    for chunk, embedding in rows:
        try:
            np1 = np.frombuffer(embedding, dtype=np.float32)
            tensor1 = torch.tensor(np1)
            cos1 = cosine_similarity(query_embed.reshape(1,-1), np1.reshape(1, -1))
            similarities.append((chunk, cos1))
        except:
            continue
    return sorted(similarities, key=lambda x: -x[1])[:top_n]

