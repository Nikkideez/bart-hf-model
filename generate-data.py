
import datetime
import os
import argparse
from dotenv import load_dotenv
from process_data import load_data
from test_generator import generate_test_dataset, replace_phrases_with_random_words, printCompare, write_datasetDict_to_txt, random_words, unique_characters, polysemous_words
from datasets import DatasetDict
import string

"""
    # You can use this script to generate a dataset
    # Just specificy the data path
    # If you don't have you data in a csv format like the file in this dir, then look at process_data
"""
""" #### Set variables """


load_dotenv()
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
target_dir = "./dump"
# dataset_format = None
dataset_path ="./data/CECW-en-ltl-dataset(combined).csv"
dataset_format ="csv"
#dataset_path = {'train': './data/hf-data/train/0000.parquet', 'test': './data/hf-data/test/0000.parquet'}
#dataset_format = "parquet"
# dataset_path = "./dump/test-gen-2023-11-07_11-36-19/train_dataset.hf"
# dataset_format = "disk"
# test_dataset_path = "./data/test_dataset.hf" # set the path to None if you want to generate a new test dataset
test_dataset_path = None
seed=42

# Some parsers so you don't have to keep changing the file
parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--targetdir', default=target_dir, help='Override the default target directory to save the training and test data.')
parser.add_argument('--savedata', action='store_true', help='Specify whether to save the loaded dataset in a csv format.')
parser.add_argument('--removeunderscore', action='store_true', help='Removes underscores from the ltl dataset.')
parser.add_argument('--nosplit', action='store_true', help='Whether to split the data')
args = parser.parse_args()

output_dir = os.path.join(args.targetdir, f"test-gen-{current_time}")
save_dataset = args.savedata
remove_underscore = args.removeunderscore
nosplit = not args.nosplit

""" #### Create the output dir if it does not exist """

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


""" #### Load Data """

_, test_subset, dataset = load_data(dataset_format, dataset_path, seed, remove_underscore=remove_underscore, split=nosplit)


if save_dataset:
    write_datasetDict_to_txt(dataset, output_dir, "train_dataset", "train")
    dataset.save_to_disk(output_dir + "/train_dataset.hf")

""" #### Generate Additional Test Data """


# Generate a new test dataset
# test_dataset = generate_test_dataset(test_subset, dataset["test"], random_words, unique_characters, polysemous_words)
# Write in txt format
# write_datasetDict_to_txt(test_dataset, output_dir, "test_dataset", "test")
# Write in an HF DataDict format if you want to easily load these again later
# test_dataset.save_to_disk(output_dir + "/test_dataset.hf")


# print(test_dataset)

print("Rare words \n")
rare_test = replace_phrases_with_random_words(test_subset, random_words=random_words)
printCompare(test_subset, rare_test)

print("\n\n")

print("Random words (Gibberish) \n")
random_test = replace_phrases_with_random_words(test_subset, characters=string.ascii_letters, min_length=5, max_length=20)
printCompare(test_subset, random_test)


print("\n\n")

print("Non-standard characters \n")
nonstd_test = replace_phrases_with_random_words(test_subset, characters=string.ascii_letters + unique_characters, min_length=5, max_length=20)
printCompare(test_subset, nonstd_test)

print("\n\n")

print("Contexualized (polysemous) words \n")
poly_test = replace_phrases_with_random_words(test_subset, random_words=polysemous_words)
printCompare(test_subset, poly_test)

test_dataset = DatasetDict({
    'original': dataset["test"],
    'rare': rare_test,
    'random': random_test,
    'nonstd': nonstd_test,
    'poly': poly_test,
})

print("\nTest dataset generated\n")
print(test_dataset, "\n")
# Write in txt format
write_datasetDict_to_txt(test_dataset, output_dir, "test_dataset", "test")
# Write in an HF DataDict format if you want to easily load these again later
test_dataset.save_to_disk(output_dir + "/test_dataset.hf")

