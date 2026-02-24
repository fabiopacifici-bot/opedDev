# Training Script
# Fine-tunes the Qwen-3 0.6B model for web development coding interview grading.

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
import os

def fine_tune_qwen3(dataset_path, model_name, output_dir):
    """Fine-tunes the Qwen-3 model."""

    # Load the processed dataset
    print("Loading processed dataset...")
    dataset = load_from_disk(dataset_path)

    # Load the pretrained Qwen-3 model and tokenizer
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize the dataset
    print("Tokenizing dataset...")
    def preprocess_function(examples):
        return tokenizer(examples['prompt'], text_pair=examples['completion'], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'completion'])

    # Define training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    print("Saving fine-tuned model...")
    trainer.save_model(output_dir)
    print("Model training complete. Fine-tuned model saved to:", output_dir)

if __name__ == "__main__":
    dataset_path = "../datasets/processed_webdev_coding"
    model_name = "Qwen/Qwen-3-0.6B"
    output_dir = "../models/qwen3_webdev"
    fine_tune_qwen3(dataset_path, model_name, output_dir)