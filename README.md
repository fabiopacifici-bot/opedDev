# opedDev

**Goal:** Implement a project based on the Hugging Face "hugging-face-model-trainer" skill, with setups for training, fine-tuning, and deploying language models. This project will leverage cloud GPUs (Hugging Face Jobs) and include workflows for SFT, DPO, GGUF conversion, and deployment tracking with Trackio.

## Features
- Model fine-tuning using TRL methods (SFT and more).
- Integration with Hugging Face Jobs infrastructure.
- Dataset validation and preprocessing utilities.
- Trackio-based monitoring and reporting.
- Model conversion to GGUF format (for local inference tools).

## Structure
This project structure will evolve as features are implemented:
```
opedDev/
├── scripts/            # Training and preprocessing scripts
├── references/         # Guidelines, instructions, and skill templates
├── datasets/           # Placeholder to organize datasets
├── models/             # Placeholder for saved models
├── logs/               # Monitoring and training log files
├── README.md           # Project documentation (this file)
```

## Next Steps
- Populate the `scripts/` folder with training and preprocessing templates.
- Install necessary Python dependencies in a virtual environment.
- Configure the required Hugging Face API credentials.
- Validate dataset handling and submission workflows.
- Test the environment by initiating a small-scale fine-tuning job.

Stay tuned as the project evolves!