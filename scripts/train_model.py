# Training Script
# Fine-tunes the Qwen-3 0.6B model for web development coding interview grading.

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
import os

# ensure logging is configured so Trainer/transformers emit to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def fine_tune_qwen3(dataset_path, model_name, output_dir):
    """Fine-tunes the Qwen-3 model."""

    # Load the processed dataset
    print("Loading processed dataset...")
    dataset = load_from_disk(dataset_path)

    # Load the pretrained Qwen-3 model and tokenizer
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Set pad token when missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with device mapping if possible
    import torch
    try:
        dtype = torch.float16 if torch.cuda.is_available() else None
    except Exception:
        dtype = None

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto', torch_dtype=dtype)
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # Tokenize the dataset
    print("Tokenizing dataset...")
    def preprocess_function(examples):
        prompts = examples['prompt']
        completions = examples['completion']
        # tokenize prompts without special tokens to get prompt lengths
        tokenized_prompts = tokenizer(prompts, add_special_tokens=False)
        # tokenize full input (prompt + completion) with padding/truncation
        full_texts = [p + c for p, c in zip(prompts, completions)]
        tokenized_full = tokenizer(full_texts, padding='max_length', truncation=True, max_length=256)

        labels = []
        for i in range(len(tokenized_full['input_ids'])):
            input_ids = tokenized_full['input_ids'][i]
            prompt_len = len(tokenized_prompts['input_ids'][i])
            lab = input_ids.copy()
            # mask prompt tokens
            for j in range(prompt_len):
                if j < len(lab):
                    lab[j] = -100
            labels.append(lab)

        tokenized_full['labels'] = labels
        return tokenized_full

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'completion'], load_from_cache_file=False)

    # Define training parameters (use env vars or defaults)
    learning_rate = float(os.environ.get('LEARNING_RATE', 2e-5))
    per_device_train_batch_size = int(os.environ.get('PER_DEVICE_TRAIN_BATCH_SIZE', 2))
    per_device_eval_batch_size = int(os.environ.get('PER_DEVICE_EVAL_BATCH_SIZE', 2))
    num_train_epochs = int(os.environ.get('NUM_TRAIN_EPOCHS', 3))
    logging_strategy = os.environ.get('LOGGING_STRATEGY', 'epoch')
    eval_strategy = os.environ.get('EVAL_STRATEGY', 'epoch')
    save_strategy = os.environ.get('SAVE_STRATEGY', 'epoch')
    save_total_limit = int(os.environ.get('SAVE_TOTAL_LIMIT', 3))

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy=logging_strategy,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )

    # Train the model (support resume via RESUME_FROM env var)
    print("Starting training...")
    resume = os.environ.get('RESUME_FROM')
    if resume:
        print("Resuming from checkpoint:", resume)
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    # Save the model
    print("Saving fine-tuned model...")
    trainer.save_model(output_dir)
    print("Model training complete. Fine-tuned model saved to:", output_dir)

if __name__ == "__main__":
    dataset_path = "../datasets/processed_webdev_coding"
    model_name = "Qwen/Qwen3-0.6B"
    output_dir = "../models/qwen3_webdev"
    # call training with small epochs/batch size
    fine_tune_qwen3(dataset_path, model_name, output_dir)