import wandb
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultFlowCallback
from process_data import *
from metrics import *
from test_generator import *

# same as the tain-model.py, just skips the training
""" #### Set variables """

load_dotenv()
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# target_dir = "/data/ndeo/bart/hypersearch"
target_dir = "./dump"
output_dir = f"{target_dir}/inference_{current_time}/"
dataset_format="csv"
dataset_path="./data/CECW-en-ltl-dataset(combined).csv" # Test datadata which new test sets are also generated with if test_dataset_path = None
test_dataset_path="./data/test_dataset.hf" # set the path to None if you want to generate a new test dataset
checkpoint = "./dump/run_2023-11-02_22-59-23/checkpoint-6075" # Loading a trained model
seed=42
report_to = "none"

""" #### Create the output dir if it does not exist """

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

""" #### Log into WANDB """
# if you set report_to="wanbd, then results are logs to wandb"
if report_to != "none":
    # Note: Environment variables are loaded from .env through load_dotenv()
    # Just make sure its in the same directory
    wandb.login()

""" #### Load and Preprocess Data """

test_dataset, dataset = load_data(dataset_format, dataset_path, seed, test_dataset_path=test_dataset_path)

tokenized_dataset, data_collator, tokenizer = preprocess_data(dataset, checkpoint)


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


""" ## Load Model """

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

print(model.config)


training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=0.00001624749612285061,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.4,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=1,
    load_best_model_at_end=True,
    save_total_limit=3,
    metric_for_best_model="eval_spot_acc",
    predict_with_generate=True,
    warmup_ratio=0.05,
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

# trainer.train()

""" # Predict test dataset """
# test_predictions = trainer.predict(tokenized_dataset["test"])
# print(test_predictions.metrics)
# write_test_metrics_to_csv(test_predictions.metrics, output_dir)

# tokenized_test["original"] = tokenized_dataset["test"]

evaluate_datadict(tokenized_test, trainer, output_dir, tokenizer)
