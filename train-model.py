
import wandb
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultFlowCallback
from process_data import *
from metrics import *

""" #### Set variables """

load_dotenv()
output_dir="/home/nikx/Documents/Projects/capstone/bart/model"
epochs=int(input("Enter the number of epochs to train: "))
dataset_format="csv"
dataset_path="/home/nikx/Downloads/CECW-en-ltl-dataset(combined).csv"
checkpoint = "facebook/bart-large"
seed=42

""" #### Log into WANDB """
# Note: Environment variables are loaded from .env through load_dotenv()
# Just make sure its in the same directory
wandb.login()

""" #### Load and Preprocess Data"""

dataset = load_data(dataset_format, dataset_path, seed)

tokenized_dataset, data_collator, tokenizer = preprocess_data(dataset, checkpoint)

""" ## Model """

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dropout=0.25)


print(model.config)


""" # Train the model """

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=0.00001624749612285061,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.4,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    logging_steps=1,
    report_to="wandb"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[DefaultFlowCallback,CSVLoggerCallback(output_dir)]
)

trainer.train()

# Predit test dataset
test_predictions = trainer.predict(tokenized_dataset["test"])
print(test_predictions)
