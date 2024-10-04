import os
import argparse
import yaml
from .util import upload_dataframe_to_s3, upload_text_to_s3, download_CSV_to_dataframe

import numpy as np
import pandas as pd

from langchain.evaluation import load_evaluator
from evaluate import load

def load_yaml(yaml_file_path):
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The file '{yaml_file_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file '{yaml_file_path}'. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading '{yaml_file_path}': {e}")

def generate_text_file(df, save_path, eval_method, columns:list):
    # Prepare the content to be written
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    content = f"Evaluation method: {eval_method}\n"
    for col in columns:
        if col in df.columns:
            mean_value = df[col].mean()
            content += f"Mean of {col}: {mean_value}\n"
        else:
            content += f"Warning: Column '{col}' not found in DataFrame.\n"

    # Write the content to a text file
    try:
        with open(save_path, 'w') as file:
            file.write(content)
        print(f"Mean distances have been written to {save_path}")
    except Exception as e:
        print(f"Error writing to file '{save_path}': {e}")

def main(yaml_file_path):

    ## LOAD config file
    config = load_yaml(yaml_file_path)
    
    ## LOAD csv file
    if not config.get("csv_file_path").lower().endswith('.csv'):
        print("Error: The provided file is not a CSV file.")
        return
    df = download_CSV_to_dataframe(config.get("bucket_name"),config.get("csv_file_path"),config.get("aws_access_key_id"),config.get("aws_secret_access_key"),config.get("aws_region"))

    ## EVALUATE using edit distance method
    if config.get("eval_method") == 'edit_distance':
        # load evaluator
        evaluator = load_evaluator("string_distance")

        def evaluate_distance(row, reference_col, gen_col):
            return evaluator.evaluate_strings(reference=row[reference_col], modell_response=row[gen_col])['score']
        # evaluate
        df['distance_gen'] = df.apply(lambda row: evaluate_distance(row, config.get("reference_col"), config.get("modell_response_col")), axis=1)
        df['distance_gen_dfl'] = df.apply(lambda row: evaluate_distance(row, config.get("reference_col"), config.get("modelr_response_col")), axis=1)
        
        # write summary output file
        upload_text_to_s3(df, config.get("bucket_name"), config.get("aws_access_key_id"), config.get("aws_secret_access_key"),config.get("aws_region"), config.get("text_save_path"), config.get("eval_method"), ['modell_distance_gen', 'modelr_distance_gen'])
    
    elif config.get("eval_method") == 'bert_score':
        bertscore = load("bertscore")
        result = bertscore.compute(predictions=df[config.get("modell_response_col")], references=df[config.get("reference_col")],lang="en")
        df['BERTscore_precision'] = result['precision']
        df['BERTscore_recall'] = result['recall']
        df['BERTscore_F1'] = result['f1']

        result_dfl = bertscore.compute(predictions=df[config.get("modelr_response_col")], references=df[config.get("reference_col")],lang="en")
        df['dfl_BERTscore_precision'] = result_dfl['precision']
        df['dfl_BERTscore_recall'] = result_dfl['recall']
        df['dfl_BERTscore_F1'] = result_dfl['f1']

        # write summary output file
        cols = ['BERTscore_precision', 'BERTscore_recall', 'BERTscore_F1', 'dfl_BERTscore_precision', 'dfl_BERTscore_recall', 'dfl_BERTscore_F1']
        upload_text_to_s3(df, config.get("bucket_name"), config.get("aws_access_key_id"), config.get("aws_secret_access_key"),config.get("aws_region"), config.get("text_save_path"), config.get("eval_method"), cols)
    else:
        print("Eval method not yet implemented")
    
    # save final dataframe
    upload_dataframe_to_s3(df,  config.get("bucket_name"), config.get("aws_access_key_id"), config.get("aws_secret_access_key"),config.get("aws_region"),config.get("save_path"))
    df.to_csv(config.get("save_path"), index=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DFL output to non-DFL output')
    parser.add_argument('yaml_file_path', type=str, help='The path to the YAML configuration file.')
    args = parser.parse_args()

    main(args.yaml_file_path)