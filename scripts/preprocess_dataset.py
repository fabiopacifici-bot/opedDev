# Preprocessing Script
# Converts dataset into a format suitable for fine-tuning

import json
from datasets import Dataset

def preprocess_dataset(input_path, output_path):
    """
    Reads the dataset, processes entries, and saves it in a fine-tuning-compatible structure.

    Args:
        input_path (str): Path to the raw dataset JSON file.
        output_path (str): Path to save the processed dataset file.

    Returns:
        None
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        processed_data = []
        print(f"Processing {len(data)} entries from dataset...")

        for item in data:
            correctness = item.get('correctness', 'Model should check correctness, clarity, and concision.')
            entry = {
                "prompt": f"{item['category']} Question: {item['question']}\nAnswer:",
                "completion": f" {item['example_answer']}\nEvaluation: {correctness}"
            }
            processed_data.append(entry)

        # Convert processed data to Hugging Face Dataset format and save
        hf_dataset = Dataset.from_list(processed_data)
        hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
        hf_dataset.save_to_disk(output_path)

        print("Preprocessing complete. Dataset saved to:", output_path)
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    input_dataset_path = "/home/pacificDev/.openclaw/workspace/repositories/opedDev/datasets/webdev-coding-interview-expanded.json"
    output_dataset_path = "../datasets/processed_webdev_coding"
    preprocess_dataset(input_dataset_path, output_dataset_path)