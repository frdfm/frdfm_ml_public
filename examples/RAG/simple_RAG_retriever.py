from frdfmml.utils.RAG_utils import retrieve_similar
from transformers import AutoTokenizer, AutoModel


seq_size = 32
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

query = "Is Moon surface sandy or rocky?"
retrieved = retrieve_similar(model, tokenizer, seq_size, query, top_n=5)
enriched_prompt = query + "\n\nUse following context for your answer.\n\n" + "\n".join([f'Score: {r[1][0][0]:6.4f} --> {r[0]}' for r in retrieved])
print(enriched_prompt)
