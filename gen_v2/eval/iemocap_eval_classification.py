import os
import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import re
import numpy as np
from scipy.stats import pearsonr
import ast

def parse_text(input_text, file_path, error_files):
    parsed_data = {}

    input_text = input_text.lower()
    input_text = input_text.replace('**', '')
    input_text = re.sub(r"label\s*:\s*", "label:", input_text)
    input_text = re.sub(r"confidence score\s*:\s*", "confidence score:", input_text)

    # Extract Label
    label_match = re.search(r"label:\s*(\{.*?\}|\w+)", input_text, re.DOTALL)
    if label_match:
        label_value = label_match.group(1)
        try:
            parsed_data['Label'] = ast.literal_eval(label_value)
        except (ValueError, SyntaxError):
            parsed_data['Label'] = label_value.strip().strip("'\"")

    # Extract Confidence Score
    try:
        confidence_score_match = re.search(r"confidence score:\s*([\d.]+)", input_text)
        if confidence_score_match:
            parsed_data['Confidence Score'] = float(confidence_score_match.group(1))
        else:
            raise ValueError("Confidence Score missing or invalid")
    except ValueError:
        error_files.append(os.path.basename(file_path))
        return None

    # Extract TrueAnswer
    true_answer_match = re.search(r"trueanswer:\s*(\{.*?\}|\w+)", input_text, re.DOTALL)
    if true_answer_match:
        try:
            parsed_data['TrueAnswer'] = ast.literal_eval(true_answer_match.group(1))
        except (ValueError, SyntaxError):
            parsed_data['TrueAnswer'] = true_answer_match.group(1).strip().strip("'\"")

    # Extract TrueLabellist
    true_label_list_match = re.search(r"truelabellist:\s*(\{.*?\}|\[.*?\])", input_text, re.DOTALL)
    if true_label_list_match:
        try:
            parsed_data['TrueLabellist'] = ast.literal_eval(true_label_list_match.group(1))
        except (ValueError, SyntaxError):
            parsed_data['TrueLabellist'] = [label.strip().strip("'\"") for label in true_label_list_match.group(1).strip('[]').split(',')]

    return parsed_data

def parser_txt(file_path, error_files):
    with open(file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()

    parsed_data = parse_text(input_text, file_path, error_files)
    return parsed_data

def process_result_files(folder_path):
    all_files = os.listdir(folder_path)
    result_files = [file for file in all_files if 'answer' in file and file.endswith('.txt')]
    return result_files

def calculate_correlations(confidence_scores, accuracies):
    accuracy_corr, _ = pearsonr(confidence_scores, accuracies)
    return accuracy_corr

def save_results(folder_path, model, processed_count, failed_count, f1_macro, f1_weighted, accuracy, avg_confidence, accuracy_corr, error_files):
    results_file = os.path.join(folder_path, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Model: {model}\n")
        f.write(f"Processed files: {processed_count}\n")
        f.write(f"Failed files: {failed_count}\n")
        if error_files:
            f.write(f"Error processing files: {error_files}\n")
        f.write(f"F1-macro score: {f1_macro}\n")
        f.write(f"F1-weighted score: {f1_weighted}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Average Confidence Score: {avg_confidence:.2f}\n")
        f.write(f"Correlation between Confidence Score and Accuracy: {accuracy_corr:.4f}\n")

def main(args):
    path = os.path.join(args.base_folder_path, args.folder_path)
    models = ['Gemini', 'Sonnet', 'GPT4o']

    for model in models:
        result_list = []
        label_list = []
        confidence_scores = []
        valid_confidence_scores = []
        valid_accuracies = []
        f1_scores = []
        accuracies = []
        processed_count = 0
        failed_count = 0
        error_files = []
        folder_path = os.path.join(path, model)
        files = process_result_files(folder_path)

        for answer_path in tqdm(files, desc=f"Processing {model} files"):
            labels = parser_txt(os.path.join(folder_path, answer_path), error_files)
            if labels is not None:
                try:
                    label_dict = labels['Label']
                    true_label_dict = labels['TrueAnswer']
                    label_list_ = labels['TrueLabellist']
                    confidence_score = labels['Confidence Score']

                    for key in label_dict:
                        predicted_label = int(label_list_[label_dict[key]])
                        true_label_index = int(label_list_[true_label_dict[key]])

                        result_list.append(predicted_label)
                        label_list.append(true_label_index)

                        # Calculate F1-score and accuracy for each prediction
                        f1_score_value = f1_score([true_label_index], [predicted_label], average='macro')
                        accuracy_value = accuracy_score([true_label_index], [predicted_label])

                        f1_scores.append(f1_score_value)
                        accuracies.append(accuracy_value)

                        valid_confidence_scores.append(confidence_score)
                        valid_accuracies.append(accuracy_value)

                    confidence_scores.append(confidence_score)
                    processed_count += 1

                except Exception as e:
                    error_files.append(answer_path)
                    failed_count += 1
            else:
                failed_count += 1

        if processed_count > 0:
            f1_macro = f1_score(label_list, result_list, average='macro')
            f1_weighted = f1_score(label_list, result_list, average='weighted')
            accuracy = accuracy_score(label_list, result_list)
            avg_confidence = np.mean(confidence_scores)

            try:
                accuracy_corr = calculate_correlations(valid_confidence_scores, valid_accuracies)
            except ValueError:
                accuracy_corr = None
        else:
            f1_macro = f1_weighted = accuracy = avg_confidence = accuracy_corr = 0.0

        error_files_str = f"[{','.join(error_files)}]" if error_files else "None"
        save_results(folder_path, model, processed_count, failed_count, f1_macro, f1_weighted, accuracy, avg_confidence, accuracy_corr, error_files_str)

        print(f"Model: {model}")
        print(f"Processed files: {processed_count}")
        print(f"Failed files: {failed_count}")
        if error_files:
            print(f"Error processing files: {error_files_str}")
        print(f"F1-macro score: {f1_macro}")
        print(f"F1-weighted score: {f1_weighted}")
        print(f"Accuracy: {accuracy}")
        print(f"Average Confidence Score: {avg_confidence:.2f}")
        print(f"Correlation between Confidence Score and Accuracy: {accuracy_corr:.4f}")
        print('-' * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process result files and calculate metrics.')

    parser.add_argument('--base_folder_path', type=str, required=False,
                        default='../results/iemocap/Classification',
                        help='Base Path to the folder containing result files')
    parser.add_argument('--folder_path', type=str, required=False,
                        default='SI-persona-expert_TQ-label_def-iemocap_defX_PS-iemocap-none_CT-iemocap_OI-iemocap_shot-0',
                        help='Path to the folder containing result files')

    args = parser.parse_args()
    main(args)
