# Dataset Validation Script
# Validates the structure and content of the dataset before training

import json

def validate_dataset(file_path):
    """
    Validates a dataset JSON file for required fields and structure.

    Args:
        file_path (str): Path to the dataset JSON file.

    Returns:
        None
    """
    required_fields = {"category", "question", "example_answer", "evaluation_criteria", "correctness"}

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        print(f"Validating dataset at: {file_path}\n")

        for idx, entry in enumerate(data):
            missing_fields = required_fields - entry.keys()
            if missing_fields:
                print(f"Entry {idx + 1} is missing fields: {missing_fields}\n")
            else:
                print(f"Entry {idx + 1}: All required fields are present.\n")

        print("\nValidation complete. Review any issues listed above.")

    except json.JSONDecodeError as e:
        print(f"Error reading the dataset: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    dataset_path = "../datasets/webdev-coding-interview.json"
    validate_dataset(dataset_path)