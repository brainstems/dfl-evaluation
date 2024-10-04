# dfl-evaluation

Goal: Evaluate dfl LLM finetuned model for generating ingredient lists based on menu item names.

## Evaluation method 1: Edit/ Levenshtein distance
The Edit distance measures the number of character-level changes (insertions, deletions, substitutions) needed to transform one string into another. This method provides a purely statistical approach to evaluating the similarity between the predicted and target ingredient lists.

## Evalution method 2: BERT score
BERTScore leverages contextualized word embeddings from pre-trained language models like BERT to assess the similarity between sentences or word sequences. It provides a more semantic approach to evaluation, considering the context and meaning of words, which is particularly useful for comparing natural language phrases.


### Running Docker
```docker
docker run -v /path/to/your/config.yaml:/app/config.yaml \
           -v /path/to/your/config_bert.yaml:/app/config_bert.yaml \
           your_image_name
```

Example config.yaml and config_bert.yaml files are provided in the repo.
Expects input_file_path (On S3) to have at least the following columns:
reference_col -> true labels
modell_response_col -> left model predictions
modelr_response_col -> right model predictions

### Example: menu item -> ingredients

reference_col: 'real' -> real ingredients from recipeNLG
modell_response_col: 'response' -> predicted ingredients from OppAI finetuned model
modelr_response_col: 'dfl_response' -> predicted ingredients from dfl finetuned model

In this example, modell_response_col and modelr_response_col where generated with prompts. However, prompt is needed for evaluation.
