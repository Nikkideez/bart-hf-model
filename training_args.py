from transformers import Seq2SeqTrainingArguments

# Define model specific training params here
# Should override any duplicate keys with base_args
model_specific_args = {
    'facebook/bart-large': {
        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,
        'learning_rate': 0.00002629697216493378,
        'weight_decay': 0.4,
        'gradient_accumulation_steps': 1,
    },
    'facebook/bart-base': {
        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,
        'learning_rate': 0.00007563606142800364,
        'weight_decay': 0.3,
        'gradient_accumulation_steps': 1,
    },
    't5-base': {
        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,
        'learning_rate': 0.00007563606142800364,
        'weight_decay': 0.3,
        'gradient_accumulation_steps': 1,
    }
}

def get_training_args(model_type, output_dir, epochs, report_to):
    base_args = {
        'output_dir': output_dir,
        'save_strategy': "epoch",
        'evaluation_strategy': "epoch",
        'logging_strategy': "epoch",
        'num_train_epochs': epochs,
        'load_best_model_at_end': True,
        'save_total_limit': 10,
        'metric_for_best_model': "eval_loss",
        'predict_with_generate': True,
        'generation_max_length': 128,
        'warmup_ratio': 0.05,
        'fp16': True,
        'push_to_hub': False,
        'logging_steps': 1,
        'report_to': report_to
    }


    # Merge the base_args with the model_specific_args for the given model_type
    return Seq2SeqTrainingArguments(**{**base_args, **model_specific_args[model_type]})
