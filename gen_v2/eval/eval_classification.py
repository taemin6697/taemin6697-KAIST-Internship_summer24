import os
import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import re
import numpy as np
from scipy.stats import pearsonr

def parse_text(input_text, file_path, error_files):
    parsed_data = {}

    input_text = input_text.lower()
    input_text = input_text.replace('*', '')
    input_text = input_text.replace('**', '')
    input_text = input_text.replace(' & ', '')
    input_text = input_text.replace('#', '')
    input_text = input_text.replace('#', '')
    input_text = input_text.replace("'", '')
    input_text = input_text.replace('"', '')
    input_text = input_text.replace('<', '')
    input_text = input_text.replace('>', '')



    input_text = re.sub(r"label\s*:\s*", "label:", input_text)
    input_text = re.sub(r"confidence score\s*:\s*", "confidence score:", input_text)

    label_match = re.search(r"label:\s*(\[.*?\]|\w+)", input_text)
    if label_match:
        label_value = label_match.group(1)
        if label_value.startswith('['):
            parsed_data['Label'] = [item.strip().strip("'\"") for item in label_value.strip('[]').split(',')]
        else:
            parsed_data['Label'] = label_value

    try:
        confidence_score_match = re.search(r"confidence score:\s*([\d.]+)", input_text)
        if confidence_score_match:
            parsed_data['Confidence Score'] = float(confidence_score_match.group(1))
        else:
            raise ValueError("Confidence Score missing or invalid")
    except ValueError:
        error_files.append(os.path.basename(file_path))
        return None

    true_answer_match = re.search(r"trueanswer:\s*(\[[^\]]+\]|\w+)", input_text)
    if true_answer_match:
        true_answer_value = true_answer_match.group(1)
        if true_answer_value.startswith('['):
            parsed_data['TrueAnswer'] = [item.strip().strip("'\"") for item in true_answer_value.strip('[]').split(',')][0]
        else:
            parsed_data['TrueAnswer'] = true_answer_value

    true_label_list_match = re.search(r"truelabellist:\[(.*?)\]", input_text)
    if true_label_list_match:
        parsed_data['TrueLabellist'] = [label.strip().strip("'\"") for label in true_label_list_match.group(1).split(',')]

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

def save_results(folder_path, models, aggregated_results):
    results_file = os.path.join(folder_path, 'aggregated_results.txt')
    with open(results_file, 'w') as f:
        for model, result in aggregated_results.items():
            f.write(f"Model: {model}\n")
            f.write(f"F1-macro: {result['f1_macro']:.4f}\n")
            f.write(f"ACC: {result['accuracy']:.4f}\n")
            f.write(f"F1-weight: {result['f1_weighted']:.4f}\n")
            f.write(f"Avg Confidence Score: {result['avg_confidence']:.4f}\n")
            f.write(f"Correlation between Confidence Score and ACC: {result['accuracy_corr']:.4f}\n")
            f.write(f"Processed files: {result['processed_count']}\n")
            f.write(f"Failed files: {result['failed_count']}\n")
            if result['error_files']:
                f.write(f"Error processing files: {result['error_files']}\n")
            f.write('-' * 50 + '\n')

def main(args):
    path = os.path.join(args.base_folder_path, args.folder_path)
    models = ['GPT4o','Gemini','Ollama_Gemma','Ollama_Qwen']
    #models = ['Ollama_Q']
    #models = ['Ollama_Mistral']#['GPT4o', 'Gemini','Llama','Qwen']
    aggregated_results = {}

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
                    label = labels['Label']
                    if isinstance(label, list):
                        label = label[0]
                    true_label = labels['TrueAnswer']
                    label_list_ = labels['TrueLabellist']
                    confidence_score = labels['Confidence Score']
                    label_list_dict = {label: i for i, label in enumerate(label_list_)}

                    predicted_label = int(label_list_dict[label])
                    true_label_index = int(label_list_dict[true_label])

                    result_list.append(predicted_label)
                    label_list.append(true_label_index)

                    # Calculate F1-score and accuracy for each prediction
                    f1_score_value = f1_score([true_label_index], [predicted_label], average='macro')
                    accuracy_value = accuracy_score([true_label_index], [predicted_label])

                    f1_scores.append(f1_score_value)
                    accuracies.append(accuracy_value)

                    # Add the confidence score only if an accuracy was calculated
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

            # Calculate correlations between confidence score and valid performance metrics
            try:
                accuracy_corr = calculate_correlations(valid_confidence_scores, valid_accuracies)
            except ValueError as ve:
                accuracy_corr = None
        else:
            f1_macro = f1_weighted = accuracy = avg_confidence = accuracy_corr = 0.0

        aggregated_results[model] = {
            'processed_count': processed_count,
            'failed_count': failed_count,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'accuracy_corr': accuracy_corr,
            'error_files': f"[{','.join(error_files)}]" if error_files else "None"
        }

        print(f"Model: {model}")
        print(f"F1-macro: {f1_macro:.4f}")
        print(f"F1-weight: {f1_weighted:.4f}")
        print(f"ACC: {accuracy:.4f}")
        print(f"Avg Confidence Score: {avg_confidence:.4f}")
        print(f"Correlation between Confidence Score and ACC: {accuracy_corr:.4f}")
        print(f"Processed files: {processed_count}")
        print(f"Failed files: {failed_count}")
        if error_files:
            print(f"Error processing files: {aggregated_results[model]['error_files']}")
        print('-' * 50)

    save_results(path, models, aggregated_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process result files and calculate metrics.')

    parser.add_argument('--base_folder_path', type=str, required=False,
                        default='../results/sdcnl/Classification',
                        help='Base Path to the folder containing result files')

    args = parser.parse_args()

    # Convert base_folder_path to absolute path to avoid issues
    base_folder_path = os.path.abspath(args.base_folder_path)

    # Get all subfolders in the base folder path
    first_level_subfolders = [os.path.join(base_folder_path, f.name) for f in os.scandir(base_folder_path) if f.is_dir()]

    for first_level_folder in first_level_subfolders:
        # Get all subfolders one level deeper
        second_level_subfolders = [os.path.join(first_level_folder, f.name) for f in os.scandir(first_level_folder) if f.is_dir()]

        for second_level_folder in second_level_subfolders:
            args.folder_path = second_level_folder  # Set the folder path to the current sub-subfolder
            print(f"Processing folder: {args.folder_path}")
            main(args)