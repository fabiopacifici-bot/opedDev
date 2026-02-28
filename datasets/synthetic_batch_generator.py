import json
from pathlib import Path
import random

# Configuration
INPUT_PATH = 'repositories/opedDev/datasets/webdev-coding-interview-expanded.json'
OUTPUT_PATH = 'repositories/opedDev/datasets/webdev-coding-interview-expanded.json'
TOTAL_NUMBER_OF_EXAMPLES = 5000

# Categories and evaluation criteria
CATEGORY_OPTIONS = ['HTML', 'CSS', 'JavaScript', 'Node.js', 'Express', 'React', 'MySQL']
EVAL_CRITERIA = [
    {"aspect": "Correctness", "weight": 0.6},
    {"aspect": "Clarity", "weight": 0.25},
    {"aspect": "Conciseness", "weight": 0.15},
]

# Load existing dataset (if any)
print("Loading dataset...")
original_dataset = []
if Path(INPUT_PATH).exists():
    with open(INPUT_PATH, 'r') as infile:
        original_dataset = json.load(infile)
print(f"Loaded {len(original_dataset)} items from original dataset.")

# Calculate how many examples need to be generated
remaining_examples = max(0, TOTAL_NUMBER_OF_EXAMPLES - len(original_dataset))
print(f"Need to generate {remaining_examples} additional examples.")

# Generate synthetic examples
def generate_example(example_id):
    category = random.choice(CATEGORY_OPTIONS)
    question = f"[Synthetic] Question {example_id} for {category}: {random.choice(['debugging', 'testing', 'security practices'])}"
    example_answer = f"[Answer {category}] This is a synthetic answer example for the question posed."
    return {
        "category": category,
        "question": question,
        "example_answer": example_answer,
        "evaluation_criteria": EVAL_CRITERIA
    }

synthetic_examples = [
    generate_example(len(original_dataset) + i + 1)
    for i in range(remaining_examples)
]

# Append synthetic data to original dataset
print(f"Appending {len(synthetic_examples)} synthetic examples...")
original_dataset.extend(synthetic_examples)

# Save to output
print("Saving updated dataset...")
with open(OUTPUT_PATH, 'w') as outfile:
    json.dump(original_dataset, outfile, indent=4)
print(f"Saved dataset with {len(original_dataset)} total items at {OUTPUT_PATH}")