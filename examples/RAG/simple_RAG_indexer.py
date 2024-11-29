from frdfmml.utils.RAG_utils import store_embeddings
from transformers import AutoTokenizer, AutoModel


seq_size = 32
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

text_file = "..//data//Moon.txt"
chunks = open(text_file).read().split("\n\n")  # Chunk by paragraph
store_embeddings(model, tokenizer, seq_size, chunks)


