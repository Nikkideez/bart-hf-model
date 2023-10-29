

import wandb
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultFlowCallback
from process_data import *
from metrics import *

""" #### Set variables """

load_dotenv()
output_dir="/data/ndeo/bart/hypersearch/"
#epochs=int(input("Enter the number of epochs to train: "))
dataset_format="csv"
dataset_path="./CECW-en-ltl-dataset(combined).csv"
# checkpoint = "facebook/bart-large"
checkpoint = "facebook/bart-base"
seed=42
projectName = 'nlp-ltl-capstone'
search_method = "bayes" 

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


""" #### Hyperparameter Search"""


sweep_config = {
    'method': search_method,
    'metric': {
       'name': 'eval_loss',
       'goal': 'maximize'
    }
}


# hyperparameter range
parameters_dict = {
    'epochs': {
        'value': 25
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'gradient_accumulation_steps': {
        'values': [1, 2, 4]
    },
}


sweep_config['parameters'] = parameters_dict


def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint
    )



sweep_id = wandb.sweep(sweep_config, project=projectName)


def train(config=None):
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config

    # set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=16,
        weight_decay=config.weight_decay,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        num_train_epochs=config.epochs,
        predict_with_generate=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=True,
        push_to_hub=False,
        logging_steps=1,
        report_to="wandb"
    )
    # define training loop
    trainer = Seq2SeqTrainer(
        #model=model,
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        callbacks=[DefaultFlowCallback,CSVLoggerCallback(output_dir)]
    )

    # start training loop
    trainer.train()


wandb.agent(sweep_id, train, count=20)






