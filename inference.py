"""
    Effectively the same as train-model.py, except you load a model that was already trained
"""
# import wandb
import datetime
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, DefaultFlowCallback
from process_data import *
from metrics import compute_metrics, CSVLoggerCallback, evaluate_datadict
from test_generator import *
from training_args import get_training_args
import argparse

""" #### Set variables """

load_dotenv()
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
target_dir = "./dump"
dataset_format ="csv"
# dataset_format = "parquet"
# dataset_format = None
# dataset_path ="./data/CECW-en-ltl-dataset(combined).csv"
dataset_path = {'train': './data/hf-data/train/0000.parquet', 'test': './data/hf-data/test/0000.parquet'}
# test_dataset_path = "./data/test_dataset.hf" # set the path to None if you want to generate a new test dataset
test_dataset_path = None
checkpoint = "facebook/bart-base"
# checkpoint = "facebook/bart-large"
seed=42
report_to = "none" # set to "wandb" if you want to report to wandb. You will need to set the .env (see the repo)
epochs = 1

# Some parsers so you don't have to keep changing the file
parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--checkpoint', default=checkpoint, help='Checkpoint to use (overwrites existing checkpoint).')
parser.add_argument('--targetdir', default=target_dir, help='Override the default target directory to save the training and test data.')
parser.add_argument('--savedata', action='store_true', help='Specify whether to save the loaded dataset in a csv format.')
parser.add_argument('--smalldataset', action='store_true', help='Makes a smaller version of the dataset using Test. Helpful for debugging large datasets.')
parser.add_argument('--removeunderscore', action='store_true', help='Removes underscores from the ltl dataset.')
parser.add_argument('--nosplit', action='store_true', help='Whether to split the data')
args = parser.parse_args()

# epochs = args.epochs if args.epochs is not None else int(input("Enter the number of epochs to train: "))
checkpoint = args.checkpoint
# report_to = args.reportto
save_dataset = args.savedata
smaller_dataset = args.smalldataset
output_dir = os.path.join(args.targetdir, f"inference_{current_time}")
remove_underscore = args.removeunderscore
nosplit = not args.nosplit

""" #### Create the output dir if it does not exist """

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


""" #### Load and Preprocess Data """

test_dataset, test_subset, dataset = load_data(dataset_format, dataset_path, seed, test_dataset_path=test_dataset_path, remove_underscore=remove_underscore, split=nosplit)

if smaller_dataset:
    dataset = create_smaller_dataset(dataset["test"])

tokenized_dataset, data_collator, tokenizer = preprocess_data(dataset, checkpoint)

if save_dataset:
    write_datasetDict_to_txt(dataset, output_dir, "train_dataset", "train")
    dataset.save_to_disk(output_dir + "/train_dataset.hf")

""" #### Generate Additional Test Data """

if not test_dataset:
    # Generate a new test dataset
    test_dataset = generate_test_dataset(dataset["test"], test_subset, random_words, unique_characters, polysemous_words)
    # Write in txt format
    write_datasetDict_to_txt(test_dataset, output_dir, "test_dataset", "test")
    # Write in an HF DataDict format if you want to easily load these again later
    test_dataset.save_to_disk(output_dir + "/test_dataset.hf")


print(test_dataset)
tokenized_test, _,_ = preprocess_data(test_dataset, checkpoint)


""" ## Model """

# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dropout=0.25)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

print(model.config)


""" # Train the model """

training_args = get_training_args(checkpoint, output_dir, epochs, report_to)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    callbacks=[DefaultFlowCallback,CSVLoggerCallback(output_dir)]
)

# trainer.train()

""" # Predict test dataset """

evaluate_datadict(tokenized_test, trainer, output_dir, tokenizer)
