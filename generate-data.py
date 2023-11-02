
import datetime
import os
from dotenv import load_dotenv
from process_data import load_data
from test_generator import generate_test_dataset, write_datasetDict, random_words, unique_characters, polysemous_words

"""
    # You can use this script to generate a dataset
    # Just specific the data path
    # If you don't have you data in a csv format like the file in this dir, then look at process_data
"""
""" #### Set variables """

load_dotenv()
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
target_dir = "./dump"
output_dir = f"{target_dir}/testdata_{current_time}/"
dataset_format="csv"
dataset_path="./CECW-en-ltl-dataset(combined).csv"
seed=42

""" #### Create the output dir if it does not exist """

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


""" #### Load Data"""

dataset = load_data(dataset_format, dataset_path, seed)

# tokenized_dataset, data_collator, tokenizer = preprocess_data(dataset, checkpoint)

""" #### Generate Additional Test Data """

test_dataset = generate_test_dataset(dataset["test"], random_words, unique_characters, polysemous_words)

# Write in txt format
write_datasetDict(test_dataset, output_dir)
# Write in an HF DataDict format if you want to easily load these again later
test_dataset.save_to_disk(output_dir + "/test_dataset.hf")

