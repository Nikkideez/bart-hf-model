""" # Functions to load and preprocess data """

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

""" #### Load Data """
def load_data(dataset_format, dataset_path, seed):
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
    
    return dataset


def preprocess_data(dataset, checkpoint):
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

    return tokenized_dataset, data_collator, tokenizer

