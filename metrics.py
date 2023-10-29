# All the metrics used for the trainer 


import os
import evaluate
import spot
import numpy as np
import os
import csv
import datetime
from transformers import TrainerCallback

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
def compute_metrics(eval_preds, tokenizer):
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
 
def write_test_metrics_to_csv(metrics, output_dir):
    # Define the CSV header and metrics keys
    CSV_HEADER = ["Loss", "Bleu", "Exact Match", "Spot Accuracy", "Gen Len", "Runtime", "Samples Per Second", "Steps Per Second"]
    METRICS_KEYS = ["test_loss", "test_bleu", "test_exact_match", "test_spot_acc", "test_gen_len", "test_runtime", "test_samples_per_second", "test_steps_per_second"]

    # Create a filename with the current time
    formatted_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filepath = os.path.join(output_dir, f"test_logs_{formatted_time}.csv")

    # Write the header and metrics to the CSV
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        writer.writerow([metrics.get(key, "") for key in METRICS_KEYS])
