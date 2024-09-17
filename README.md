# dfl-evaluation

Goal: Evaluate dfl LLM finetuned model for generating ingredient lists based on menu item names.

## Evaluation method 1: Edit/ Levenshtein distance
The Edit distance measures the number of character-level changes (insertions, deletions, substitutions) needed to transform one string into another. This method provides a purely statistical approach to evaluating the similarity between the predicted and target ingredient lists.

## Evalution method 2: BERT score
BERTScore leverages contextualized word embeddings from pre-trained language models like BERT to assess the similarity between sentences or word sequences. It provides a more semantic approach to evaluation, considering the context and meaning of words, which is particularly useful for comparing natural language phrases.
