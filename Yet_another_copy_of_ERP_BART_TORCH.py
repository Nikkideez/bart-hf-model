#!/usr/bin/env python
# coding: utf-8

# # HUGGING FACE BART TRANSLATOR (TORCH)
# I followed the instructions at https://huggingface.co/docs/transformers/tasks/translation

# from google.colab import drive
# drive.mount('/content/drive')


# #### Install Dependencies

# get_ipython().run_line_magic('pip', 'install transformers==4.33.1 datasets==2.14.5 evaluate==0.4.0 sacrebleu==2.3.1 accelerate==0.22.0 wandb==0.15.12 python-dotenv')


# %pip uninstall -y wandb


# get_ipython().run_line_magic('pip', "list | grep 'transformers\\|datasets\\|evaluate\\|sacrebleu\\|accelerate\\|wandb\\|dotenv'")



import os
import wandb
import evaluate
import spot
import numpy as np
import os
import csv
import datetime
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq,AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultFlowCallback, TrainerCallback


""" #### Set variables """

load_dotenv()
output_dir="/data/ndeo/bart"
epochs=int(input("Enter the number of epochs to train: "))
dataset_format="csv"
dataset_path="/home/ndeo/Downloads/data/og-dataset/CECW-en-ltl-dataset(combined) - Sheet1.csv"
checkpoint = "facebook/bart-large"
seed=42

""" #### Log into WANDB """

wandb.login()


""" #### Load Data """

# dataset = load_dataset("cRick/NL-to-LTL-Synthetic-Dataset")
# dataset = load_dataset("parquet", data_files={'train': '/content/drive/MyDrive/ERP/train/0000.parquet', 'test': '/content/drive/MyDrive/ERP/test/0000.parquet'})
# dataset = load_dataset("text", data_files={'ltl': '/content/drive/MyDrive/ERP/data_src_combined.txt', 'en': '/content/drive/MyDrive/ERP/data_tar_combined.txt'})
dataset = load_dataset(dataset_format, data_files=dataset_path)

print(dataset)


""" #### Split the data into Train, Valid and Test """

train_test = dataset["train"].train_test_split(test_size=0.3, seed=seed)
test_valid = train_test["test"].train_test_split(test_size=0.5, seed=seed)

dataset = DatasetDict({
    'train': train_test["train"],
    'test': train_test["test"],
    'valid': test_valid["test"],
    })

print(dataset)


""" #### Check for longest word in dataset to find max_length """
""" ###### Note this part may not be necessary - dynamic padding would be a more effective solution """

# Assuming your 2D array is named 'data'
max_length = 0

for column in dataset['train']['en']:
    column_length = len(column)
    if column_length > max_length:
        max_length = column_length

print("Maximum length:", max_length)


""" #### Preprocess/Tokenize Data """


tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "ltl"
prefix = "translate English to LTL: "


def preprocess_function(examples):
    inputs = [prefix + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)


""" #### Data Collator """


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


""" #### Metrics """

# evalpred_global = None
# result_global = None


# def write_to_txt(data, filename):
#     with open(filename, 'w') as f:
#         f.write(str(data))




metric = evaluate.combine(["sacrebleu","exact_match"])
# metric = evaluate.combine(["accuracy","sacrebleu"])



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

# custom metric to use the SPOT library to get metrics
def are_formulas_equal(preds, labels):
    if len(preds) != len(labels):
        raise ValueError("Both lists should be of the same length.")
    
    c = spot.language_containment_checker()
    results = []
    
    for pred, label in zip(preds, labels):
        try:
            f = spot.formula(pred)
            g = spot.formula(label)
            result = 1 if pred == label or c.equal(f, g) else 0
            results.append(result)
        except Exception as e:
            # print(e)
            result = 1 if pred == label else 0
            results.append(result)
    return results

# Computing metrics
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Calculate formula equality accuracy
    spot_results = are_formulas_equal(decoded_preds, decoded_labels)
    spot_accuracy = sum(spot_results) / len(spot_results)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"], "exact_match": result["exact_match"], "spot_acc": spot_accuracy}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Write metrics to a CSV


class CSVLoggerCallback(TrainerCallback):
    CSV_HEADER = ["Epoch", "Training Loss", "Validation Loss", "Bleu", "Exact Match", "Spot Accuracy", "Gen Len"]
    VALIDATION_METRICS = [
        ("Validation Loss", "eval_loss"),
        ("Bleu", "eval_bleu"),
        ("Exact Match", "eval_exact_match"),
        ("Spot Accuracy", "eval_spot_acc"),
        ("Gen Len", "eval_gen_len")
    ]

    def __init__(self, output_dir):
        super().__init__()
        formatted_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.filepath = os.path.join(output_dir, f"training_logs_{formatted_time}.csv")

        with open(self.filepath, 'w', newline='') as file:
            csv.writer(file).writerow(self.CSV_HEADER)
        self.current_epoch_data = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            self.current_epoch_data["Training Loss"] = logs["loss"]

        for metric_name, log_key in self.VALIDATION_METRICS:
            if log_key in logs:
                self.current_epoch_data[metric_name] = logs[log_key]

        if (state.epoch).is_integer() and all(key in self.current_epoch_data for key in self.CSV_HEADER[1:]):
            with open(self.filepath, 'a', newline='') as file:
                csv.writer(file).writerow([state.epoch] + [self.current_epoch_data.get(key, "") for key in self.CSV_HEADER[1:]])
            self.current_epoch_data = {}

""" ## Model """


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dropout=0.25)


print(model.config)


""" #### Hyperparameter Search with WANDB """

# sweep_config = {
#     'method': 'random'
# }


# # hyperparameters
# parameters_dict = {
#     'epochs': {
#         'value': 10
#         },
#     'batch_size': {
#         'values': [8, 16, 32, 64]
#         },
#     'learning_rate': {
#         'distribution': 'log_uniform_values',
#         'min': 1e-5,
#         'max': 1e-3
#     },
#     'weight_decay': {
#         'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#     },
# }


# sweep_config['parameters'] = parameters_dict


# In[28]:


# def model_init():
#     return AutoModelForSeq2SeqLM.from_pretrained(
#         checkpoint
#     )


# In[29]:


# sweep_id = wandb.sweep(sweep_config, project='nlp-ltl-capstone')


# In[30]:


# def train(config=None):
#   with wandb.init(config=config):
#     # set sweep configuration
#     config = wandb.config

#     # set training arguments
#     training_args = Seq2SeqTrainingArguments(
#         output_dir="/content/drive/MyDrive/ERP",
#         learning_rate=config.learning_rate,
#         per_device_train_batch_size=config.batch_size,
#         per_device_eval_batch_size=32,
#         weight_decay=config.weight_decay,
#         save_strategy='epoch',
#         evaluation_strategy='epoch',
#         logging_strategy='epoch',
#         num_train_epochs=config.epochs,
#         predict_with_generate=True,
#         load_best_model_at_end=True,
#         fp16=True,
#         push_to_hub=False,
#         logging_steps=1,
#         report_to="wandb"
#     )
#     # define training loop
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset["train"],
#         eval_dataset=tokenized_dataset["valid"],
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#         callbacks=[DefaultFlowCallback,CSVLoggerCallback("/content/drive/MyDrive/ERP")]
#     )

#     # start training loop
#     trainer.train()


# In[31]:


# wandb.agent(sweep_id, train, count=20)


# #### Train the Model

# In[32]:


# get_ipython().run_line_magic('env', 'WANDB_DISABLED=true')


# In[33]:


training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=0.00001624749612285061,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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

# In[25]:


# text = "it is never the case that HUymGnOfeUvUnxi"
# # Answer: G(!( HUymGnOfeUvUnxi ))
# text2 = "a train has arrived involves that as a semaphore is green, in the future the bar is down"
# # Answer2: "G(( train_has_arrived ) -> G(( semaphore_is_green ) -> F( bar_is_down )))"


# In[26]:


# from transformers import pipeline

# translator = pipeline("translation", model="/content/drive/MyDrive/ERP/checkpoint-500", max_length=198)


# In[ ]:


# print(translator(text))
# print(translator(text2))


# In[ ]:


# !pip list | grep torch

