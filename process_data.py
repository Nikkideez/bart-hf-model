""" # Functions to load and preprocess data """

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BartTokenizer


""" #### Load Data """

def replace_underscores(example):
    example['ltl'] = example['ltl'].replace('_', ' ')
    return example

def load_data(dataset_format, dataset_path, seed, test_dataset_path=None, remove_underscore=False):
    # dataset = load_dataset("cRick/NL-to-LTL-Synthetic-Dataset")
    # dataset = load_dataset("parquet", data_files={'train': '/content/drive/MyDrive/ERP/train/0000.parquet', 'test': '/content/drive/MyDrive/ERP/test/0000.parquet'})
    # dataset = load_dataset("text", data_files={'ltl': '/content/drive/MyDrive/ERP/data_src_combined.txt', 'en': '/content/drive/MyDrive/ERP/data_tar_combined.txt'})
    if dataset_format == None:
        dataset = load_dataset(dataset_path)
    else:
        dataset = load_dataset(dataset_format, data_files=dataset_path)
     
    
    print(dataset)


    """ #### Split the data into Train, Valid and Test """

    if dataset_format == "csv":
        train_test = dataset["train"].train_test_split(test_size=0.3, seed=seed)
        test_valid = train_test["test"].train_test_split(test_size=0.5, seed=seed)

        dataset = DatasetDict({
            'train': train_test["train"],
            'test': test_valid["train"],
            'valid': test_valid["test"],
            })
    elif dataset_format == "parquet" or dataset_format == None:
        train_valid = dataset["train"].train_test_split(test_size=0.1, seed=seed)

        dataset["train"] = train_valid["train"]
        dataset["valid"] = train_valid["test"]
    else:
        print("Dataset format unsupported. Check load_data and adjust the code manually. Refer to https://huggingface.co/docs/datasets/loading")

    print(dataset)

    # Remove underscores
    if remove_underscore:
        print("Removing underscores")
        dataset = dataset.map(lambda examples: {'ltl': [s.replace('_', ' ') for s in examples['ltl']]}, batched=True)

    if test_dataset_path:
        test_dataset = load_from_disk(test_dataset_path)
    else:
        test_dataset = None

    return test_dataset, dataset

def create_smaller_dataset(dataset, seed=42):
    # Splits data into a new test, train and eval dataset 

    train_test = dataset.train_test_split(test_size=0.3, seed=seed)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=seed)

    dataset = DatasetDict({
        'train': train_test["train"],
        'test': test_valid["train"],
        'valid': test_valid["test"],
        })

    print("\nCreating smaller dataset \n")
    print(dataset)

    return dataset

def preprocess_data(dataset, checkpoint):
    """ #### Check for longest word in dataset to find max_length """
    """ ###### Note this part may not be necessary - dynamic padding would be a more effective solution """
    ## assuming your 2d array is named 'data'
    #max_length = 0

    #for column in dataset['train']['en']:
    #    column_length = len(column)
    #    if column_length > max_length:
    #        max_length = column_length

    #print("maximum length:", max_length)

    """ #### Preprocess/Tokenize Data """

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenizer = BartTokenizer.from_pretrained(checkpoint)

    source_lang = "en"
    target_lang = "ltl"
    prefix = "translate English to LTL: "

    def preprocess_function(examples):
        inputs = [prefix + example for example in examples[source_lang]]
        targets = [example for example in examples[target_lang]]
        model_inputs = tokenizer(inputs, text_target=targets, truncation=True)

        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    """ #### Data Collator """

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, label_pad_token_id=tokenizer.pad_token_id)

    return tokenized_dataset, data_collator, tokenizer

