#! /bin/bash

# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data logs results

# Generate data
python src/data_generator.py --num_samples 10000 --bit_length 10 --output_file train.csv
python src/data_generator.py --num_samples 1000 --bit_length 10 --output_file test.csv
