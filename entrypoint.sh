#!/bin/bash

set -e

echo "Starting the dfl_evaluation.py script..."


python3 /app/dfl_evaluation.py "config.yaml"


if [ $? -ne 0 ]; then
    echo "Error running dfl_evaluation.py with config.yaml"
    exit 1
fi


python3 /app/dfl_evaluation.py "config_bert.yaml"


if [ $? -ne 0 ]; then
    echo "Error running dfl_evaluation.py with config_bert.yaml"
    exit 1
fi

echo "Completed running the dfl_evaluation.py script."
