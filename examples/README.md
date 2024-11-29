# Examples

## Retrieval-Augmented Generation (RAG)

This is a simple, unoptimized example to generate enriched prompts using local data.

Run `RAG/simple_RAG_indexer` once to create the `local_data.db` from `data/Moon.txt`.

Run `RAG/simple_RAG_retriever` to generate enriched prompts.

## English to French translation

You need to download the en-fr translation dataset (en-fr.csv) from Kaggle website into `data/` directory first. 

https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset?resource=download

Then you can directly run `en_fr_translation/en_fr_dataloader.py` to verify batch tensor generation.

Then, you can run `en_fr_translation/en_fr_translation.py` to train a model from scratch (needs hyperparameter tuning). 