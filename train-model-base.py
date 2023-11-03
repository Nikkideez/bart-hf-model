
import wandb
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultFlowCallback
from process_data import *
from metrics import *
from test_generator import *
import argparse

""" #### Set variables """

load_dotenv()
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# target_dir = "/data/ndeo/bart/hypersearch"
target_dir = "./dump"
output_dir = f"{target_dir}/run_{current_time}/"
# dataset_format ="csv"
# dataset_format = "parquet"
dataset_format = None
# dataset_path ="./data/CECW-en-ltl-dataset(combined).csv"
# dataset_path ="./data/hf-data/unrestricted_train_dataset-140.csv"
# dataset_path = {'train': './data/hf-data/train/0000.parquet', 'test': './data/hf-data/test/0000.parquet'}
dataset_path = "cRick/NL-to-LTL-Synthetic-Dataset"
# test_dataset_path = "./data/test_dataset.hf" # set the path to None if you want to generate a new test dataset
test_dataset_path = None
#checkpoint = "facebook/bart-large"
checkpoint = "facebook/bart-base"
seed=42
report_to = "none" # set to "wandb" if you want to report to wandb. You will need to set the .env (see the repo)
# report_to = "wandb"

# Some parsers so you don't have to keep changing the file
parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--epochs', help='Number of epochs to train')
parser.add_argument('--checkpoint', help='Checkpoint to use (overwrites existing checkpoint)')
parser.add_argument('--reportto', help='Specify if the trainer will report training metrics. Value is passed into report-to argument in trainer args.')
args = parser.parse_args()

if args.epochs:
    epochs = int(args.epochs)
else:
    epochs = int(input("Enter the number of epochs to train: "))

if args.checkpoint:
    checkpoint = args.checkpoint

if args.reportto:
    report_to = args.reportto


""" #### Create the output dir if it does not exist """

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

""" #### Log into WANDB """
if report_to != "none":
    # Note: Environment variables are loaded from .env through load_dotenv()
    # Just make sure its in the same directory
    print("Logging into wanbd")
    wandb.login()

""" #### Load and Preprocess Data """

test_dataset, dataset = load_data(dataset_format, dataset_path, seed, test_dataset_path=test_dataset_path)
tokenized_dataset, data_collator, tokenizer = preprocess_data(dataset, checkpoint)

# tokenized_dir = os.path.join(output_dir, "tokenized_dataset")
# os.makedirs(tokenized_dir)
print(tokenized_dataset)
# print(tokenized_dataset["train"]["input_ids"])
# write_tok_datasetDict(tokenized_dataset, output_dir)


""" #### Generate Additional Test Data """

if not test_dataset:
    
    # Generate a new test dataset
    test_dataset = generate_test_dataset(dataset["test"], random_words, unique_characters, polysemous_words)

    # Write in txt format
    write_datasetDict(test_dataset, output_dir)
    # Write in an HF DataDict format if you want to easily load these again later
    test_dataset.save_to_disk(output_dir + "/test_dataset.hf")


print(test_dataset)
tokenized_test, _,_ = preprocess_data(test_dataset, checkpoint)


""" ## Model """

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dropout=0.20)

print(model.config)


""" # Train the model """

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=0.00007563606142800364,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.3,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=epochs,
    load_best_model_at_end=True,
    save_total_limit=3,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,
    warmup_ratio=0.05,
    generation_max_length=70,
    fp16=True,
    push_to_hub=False,
    logging_steps=1,
    report_to=report_to
)

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

trainer.train()

""" # Predict test dataset """
# test_predictions = trainer.predict(tokenized_dataset["test"])
# print(test_predictions.metrics)
# write_test_metrics_to_csv(test_predictions.metrics, output_dir)

# tokenized_test["original"] = tokenized_dataset["test"]

evaluate_datadict(tokenized_test, trainer, output_dir, tokenizer)
