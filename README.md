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

## Requirements

### Runtime
- Python 3.10+
- pip / virtualenv

### Key Dependencies
Install via `pip install -r requirements.txt` (if present) or manually:

| Package           | Purpose                                      |
|-------------------|----------------------------------------------|
| `torch`           | Deep learning framework                      |
| `transformers`    | Hugging Face model loading & training        |
| `datasets`        | Dataset loading & preprocessing              |
| `trl`             | SFT, DPO, GRPO training methods              |
| `accelerate`      | Multi-GPU / mixed-precision training         |
| `huggingface_hub` | Model pushing & pulling from HF Hub          |
| `safetensors`     | Fast, safe model serialization               |
| `tokenizers`      | Fast tokenizer support                       |

### Environment Variables
Create a `.env` file or set these in your shell:

| Variable   | Description                                              |
|------------|----------------------------------------------------------|
| `HF_TOKEN` | Hugging Face token (write access for pushing models)     |

Get your token at: https://huggingface.co/settings/tokens

## Next Steps
- Populate the `scripts/` folder with training and preprocessing templates.
- Install necessary Python dependencies in a virtual environment.
- Configure the required Hugging Face API credentials.
- Validate dataset handling and submission workflows.
- Test the environment by initiating a small-scale fine-tuning job.

Stay tuned as the project evolves!

## Model artifacts

The fine-tuned model `qwen3_webdev` is saved at `~/.openclaw/models/qwen3_webdev`. A symlink has been created in this repository at `repositories/opedDev/models/qwen3_webdev` pointing to that location.

To load the model for inference (example):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# We recommend loading the tokenizer from the original HF model and the weights from the local folder
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('models/qwen3_webdev', trust_remote_code=True, device_map='auto', torch_dtype='float16')

prompt = "def add(a,b):\n    # implement add\n"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Notes:
- We used `trust_remote_code=True` because the Qwen tokenizer/model rely on custom classes from the HF repo.
- If you want a repo-contained tokenizer, copy the tokenizer files from the HF cache (`~/.cache/huggingface/hub/...`) into `repositories/opedDev/models/qwen3_webdev`.
- Checkpoints are stored under `~/.openclaw/models/qwen3_webdev` and include `checkpoint-45` (optimizer & trainer state).



## Specs

Artifacts from the smoke test and the training plan are in .specs/:
- .specs/smoke_test_output.txt
- .specs/training/plan.md

