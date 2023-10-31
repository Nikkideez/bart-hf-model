

import wandb
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultFlowCallback
from process_data import *
from metrics import *
from test_generator import *


""" #### Set variables """

load_dotenv()
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
target_dir = "/home/nikx/capstone/dump"
output_dir = f"{target_dir}/run_{current_time}/"
dataset_format="csv"
dataset_path="./CECW-en-ltl-dataset(combined).csv"
checkpoint = "/home/nikx/capstone/dump/checkpoint-1500" # Replace this with some path to a trained checkpoint
seed=42
epochs=int(input("Enter the number of epochs to train: "))

""" #### Create the output dir if it does not exist """

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

""" #### Log into WANDB """
# Note: Environment variables are loaded from .env through load_dotenv()
# Just make sure its in the same directory
wandb.login()

""" #### Load and Preprocess Data"""

dataset = load_data(dataset_format, dataset_path, seed)

tokenized_dataset, data_collator, tokenizer = preprocess_data(dataset, checkpoint)

""" #### Generate Additional Test Data """

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
    'rare': rare_test,
    'random': random_test,
    'nonstd': nonstd_test,
    'poly': poly_test,
})

print(test_dataset)

write_datasetDict(test_dataset, output_dir)
write_array_to_file(dataset["test"]["en"], f"{output_dir}/test-original-eng.txt")
write_array_to_file(dataset["test"]["ltl"], f"{output_dir}/test-original-ltl.txt")
tokenized_test, _,_ = preprocess_data(test_dataset, checkpoint)

""" ## Model """

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dropout=0.25)

print(model.config)


""" # Train the model """

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
    report_to="wandb"
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

tokenized_test["original"] = tokenized_dataset["test"]

evaluate_datadict(tokenized_test, trainer, output_dir, tokenizer)
