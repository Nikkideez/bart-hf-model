
import datetime
import os
from dotenv import load_dotenv
from process_data import load_data
from test_generator import generate_test_dataset, replace_phrases_with_random_words, printCompare, write_datasetDict, random_words, unique_characters, polysemous_words
from datasets import DatasetDict
import string

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
dataset_path="./data/CECW-en-ltl-dataset(combined).csv"
seed=42

""" #### Create the output dir if it does not exist """

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


""" #### Load Data"""

test_dataset, dataset = load_data(dataset_format, dataset_path, seed)

# tokenized_dataset, data_collator, tokenizer = preprocess_data(dataset, checkpoint)

""" #### Generate Additional Test Data """

# test_dataset = generate_test_dataset(dataset["test"], random_words, unique_characters, polysemous_words)


print("Rare words \n")
rare_test = replace_phrases_with_random_words(dataset["test"], random_words=random_words)
printCompare(dataset["test"], rare_test)

print("\n\n")

print("Random words (Gibberish) \n")
random_test = replace_phrases_with_random_words(dataset["test"], characters=string.ascii_letters, min_length=5, max_length=20)
printCompare(dataset["test"], random_test)


print("\n\n")

print("Non-standard characters \n")
nonstd_test = replace_phrases_with_random_words(dataset["test"], characters=string.ascii_letters + unique_characters, min_length=5, max_length=20)
printCompare(dataset["test"], nonstd_test)

print("\n\n")

print("Contexualized (polysemous) words \n")
poly_test = replace_phrases_with_random_words(dataset["test"], random_words=polysemous_words)
printCompare(dataset["test"], poly_test)

test_dataset = DatasetDict({
    'original': dataset["test"],
    'rare': rare_test,
    'random': random_test,
    'nonstd': nonstd_test,
    'poly': poly_test,
})

# Write in txt format
write_datasetDict(test_dataset, output_dir)
# Write in an HF DataDict format if you want to easily load these again later
test_dataset.save_to_disk(output_dir + "/test_dataset.hf")

