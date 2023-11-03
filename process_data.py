""" # Functions to load and preprocess data """

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BartTokenizer

""" #### Load Data """
def load_data(dataset_format, dataset_path, seed, test_dataset_path=None):
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
        # train_valid = dataset["train"].train_test_split(test_size=0.1, seed=seed)

        # dataset["train"] = train_valid["train"]
        # dataset["valid"] = train_valid["test"]
        # Im using the smalller test set for just debugging
        dataset = dataset.remove_columns("pair_type") 
        train_test = dataset["test"].train_test_split(test_size=0.3, seed=seed)
        test_valid = train_test["test"].train_test_split(test_size=0.5, seed=seed)

        dataset = DatasetDict({
            'train': train_test["train"],
            'test': test_valid["train"],
            'valid': test_valid["test"],
            })

    else:
        print("Dataset format unsupported. Check load_data and adjust the code manually. Refer to https://huggingface.co/docs/datasets/loading")

    print(dataset)

    if test_dataset_path:
        test_dataset = load_from_disk(test_dataset_path)
    else:
        test_dataset = None

    return test_dataset, dataset


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


    """ #### preprocess/Tokenize Data """


    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenizer = BartTokenizer.from_pretrained(checkpoint)

    source_lang = "en"
    target_lang = "ltl"
    prefix = "translate English to LTL: "


    def preprocess_function(examples):
        #inputs = [prefix + example for example in examples[source_lang]]
        #model_inputs = tokenizer(inputs, max_length=256, padding="max_length", truncation=True)

        ## Setup the tokenizer for targets
        #with tokenizer.as_target_tokenizer():
        #    labels = tokenizer(examples[target_lang], max_length=256, padding="max_length", truncation=True)

        #model_inputs["labels"] = labels["input_ids"]
        inputs = [prefix + example for example in examples[source_lang]]
        targets = [example for example in examples[target_lang]]
        #model_inputs = tokenizer(inputs, text_target=targets, max_length=256, truncation=True)
        model_inputs = tokenizer(inputs, text_target=targets, truncation=True)

        return model_inputs


    tokenized_dataset = dataset.map(preprocess_function, batched=True)


    """ #### Data Collator """


    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, label_pad_token_id=tokenizer.pad_token_id)

    #class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
        #def __init__(self, *args, max_target_length=None, **kwargs):
            #super().__init__(*args, **kwargs)
            #self.max_target_length = max_target_length

        #def __call__(self, features, return_tensors=None):
            #if self.max_target_length is not None:
                ## Truncate the target sequences to max_target_length
                #for feature in features:
                    #target_length = min(len(feature['labels']), self.max_target_length)
                    #feature['labels'] = feature['labels'][:target_length]

            ## Now use the parent class's __call__ to do the actual data collation
            #return super().__call__(features, return_tensors=return_tensors)

    ## Define your custom max_target_length
    #max_target_length = 1000 # for example

    ## Instantiate the custom data collator
    #data_collator = CustomDataCollatorForSeq2Seq(
        #tokenizer=tokenizer,
        #model=checkpoint,
        #max_target_length=max_target_length,
        ## Add any other arguments needed for DataCollatorForSeq2Seq
    #)

    return tokenized_dataset, data_collator, tokenizer

