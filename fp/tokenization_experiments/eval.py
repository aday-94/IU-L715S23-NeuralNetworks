# BLEU for segment by segment with arguments
# Run this file from CMD/Terminal
# Example Command: python3 eval.py toke-tgt-5.txt pred_mean_rnn_aug_1000.txt

import sys
from asrtoolkit import cer

target_test = sys.argv[1]  # Test file argument
target_pred = sys.argv[2]  # MTed file argument

cer_result = cer(target_test, target_pred)
print(f"Character Error Rate: {cer_result}")

""" import sys
from asrtoolkit import cer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

target_test = sys.argv[1]  # Test file argument
target_pred = sys.argv[2]  # MTed file argument

# Calculate CER
cer_result = cer(target_test, target_pred)
print(f"Character Error Rate: {cer_result}")

# Load ground truth and predicted labels
with open(target_test, 'r') as f_test, open(target_pred, 'r') as f_pred:
    test_lines = f_test.readlines()
    pred_lines = f_pred.readlines()

# Tokenize the lines
test_tokens = [line.strip().split() for line in test_lines]
pred_tokens = [line.strip().split() for line in pred_lines]

# Flatten the lists of tokens
test_flat = [token for line in test_tokens for token in line]
pred_flat = [token for line in pred_tokens for token in line]

# Calculate F1 score, precision, recall, and accuracy
f1 = f1_score(test_flat, pred_flat, average='weighted')
precision = precision_score(test_flat, pred_flat, average='weighted')
recall = recall_score(test_flat, pred_flat, average='weighted')
accuracy = accuracy_score(test_flat, pred_flat)

print(f"F1 score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
print(f"Character Error Rate: {cer_result}") """

""" import re
import asrtoolkit
from sklearn.metrics import f1_score


def flatten_list(l):
    Flatten a nested list
    return [item for sublist in l for item in sublist]


def standardize_transcript(input_transcript, remove_nsns=False):
    #Remove non-speech noise and standardize format
    input_transcript = re.sub(r'\[.*?\]', '', input_transcript)
    input_transcript = re.sub(r'\(.*?\)', '', input_transcript)
    input_transcript = re.sub(r'\{.*?\}', '', input_transcript)
    input_transcript = re.sub(r'[^\w\s]', '', input_transcript)
    input_transcript = re.sub(r'\d', '', input_transcript)
    input_transcript = input_transcript.strip()
    if remove_nsns:
        re_tagged_nonspeech = re.compile(r'<.*?>')
        input_transcript = re.sub(re_tagged_nonspeech, ' ', input_transcript)
    return input_transcript


def cer(targets, preds):
    ref, hyp = map(
        lambda transcript: standardize_transcript(transcript, remove_nsns=True),
        [targets, preds]
    )
    return asrtoolkit.cer(ref, hyp)


def evaluate(targets, preds):
    # Flatten the target and prediction lists
    targets = flatten_list(targets)
    preds = flatten_list(preds)

    # Convert the target and prediction lists to strings
    targets_str = ' '.join(targets)
    preds_str = ' '.join(preds)

    # Check that the tokenized versions of target and prediction have the same length
    if len(targets) != len(preds):
        raise ValueError('Error: Tokenized versions of target and prediction do not have the same length.')

    # Compute CER and F1 score
    cer_result = cer(targets_str, preds_str)
    f1_result = f1_score(targets_str.split(), preds_str.split(), average='macro')

    return cer_result, f1_result """

""" import re
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(targets, preds):
    # Convert the target and predicted sequences to lists of characters
    targets = [char for char in targets.replace(" ", "")]
    preds = [char for char in preds.replace(" ", "")]

    # Compute evaluation metrics
    f1_result = f1_score(targets, preds, average="macro")
    precision_result = precision_score(targets, preds, average="macro")
    recall_result = recall_score(targets, preds, average="macro")

    return {
        "f1_score": f1_result,
        "precision": precision_result,
        "recall": recall_result,
    }

def read_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()

def evaluate(file_path_targets, file_path_preds):
    # Read the target and predicted sequences from files
    targets = read_file(file_path_targets)
    preds = read_file(file_path_preds)

    # Check if the number of sequences is the same
    if len(targets) != len(preds):
        raise ValueError("Number of targets and predictions do not match")

    # Evaluate each sequence and accumulate the results
    total_metrics = {
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    for target, pred in zip(targets, preds):
        metrics = compute_metrics(target.strip(), pred.strip())
        for metric_name in total_metrics.keys():
            total_metrics[metric_name] += metrics[metric_name]

    # Compute the average of the evaluation metrics
    num_sequences = len(targets)
    avg_metrics = {}
    for metric_name in total_metrics.keys():
        avg_metrics[metric_name] = total_metrics[metric_name] / num_sequences

    return avg_metrics

target_file_path = "toke-tgt-5.txt"
pred_file_path = "pred_mean_rnn_1000.txt"

metrics = evaluate(target_file_path, pred_file_path)
print(metrics) """