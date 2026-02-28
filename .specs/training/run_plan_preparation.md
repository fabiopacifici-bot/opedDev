Run plan: prepare dataset merging, validation, preprocessing, and training submission for qwen3_webdev

Overview
- Purpose: follow .specs/training/plan.md to incrementally fine-tune Qwen/Qwen3-0.6B for webdev coding grading in +100-example steps.
- Scope: operate only inside repositories/opedDev and subfolders. No training will be started until Olly approves.

Preflight (checks before any job)
1. Confirm next batch exists at: repositories/opedDev/datasets/incremental/<batch_dir> (e.g. batch_100_03)
2. Verify hold-out validation set located at: repositories/opedDev/datasets/processed_webdev_coding/validation.jsonl (or similar). Do NOT change it.
3. Ensure latest checkpoint is available at: repositories/opedDev/models/qwen3_webdev (symlink to ~/.openclaw/models/qwen3_webdev). If missing, note and request sync.
4. Ensure HF_TOKEN is available to Olly for job submission (we will not store it here). Job submission requires secrets={"HF_TOKEN": "$HF_TOKEN"}.

Dataset merging and validation
- Merge strategy options:
  A) On-repo merge: create merged_train.jsonl under repositories/opedDev/datasets/processed_webdev_coding by concatenating existing train + all incremental batches up to current.
  B) Trainer-side dynamic loading: keep incremental batches separate and point Trainer data loader to list of dataset paths (recommended to avoid large file commits).

Recommended (safe) workflow:
1. Run dataset inspector (fast CPU job) to validate format before any GPU job. Example hf_jobs call (assistant will run this after Olly approval to continue):

hf_jobs("uv", {
  "script": "https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py",
  "script_args": ["--dataset", "file:./repositories/opedDev/datasets/processed_webdev_coding", "--split", "train"],
  "flavor": "cpu-basic",
  "timeout": "15m"
})

2. If inspector reports NEEDS MAPPING, apply mapping code it returns. Mapping should be performed into repositories/opedDev/datasets/processed_webdev_coding_mapped/ and saved there; keep original files intact.
3. After mapping, re-run inspector until it reports READY.

Preprocessing
- Tokenization and sequence trimming will be handled inside the training script. If a separate preprocessing step is desired (to save time during job), produce a tokenized dataset under repositories/opedDev/datasets/tokenized/<batch_tag> using a CPU job or locally.

Training run plan (template - DO NOT SUBMIT WITHOUT OLly'S APPROVAL)
- Model: Qwen/Qwen3-0.6B
- Resume from: latest checkpoint under repositories/opedDev/models/qwen3_webdev (if exists)
- Hyperparameters (per plan.md defaults):
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1
  learning_rate: 2e-5 (use 1e-5 if resuming)
  num_train_epochs: 3
  fp16: True
  logging_strategy: 'epoch'
  save_strategy: 'epoch' (keep last 3)
  evaluation_strategy: 'epoch'
  seed: 42

- LoRA alternative: r=8/16, alpha=16, dropout=0.05, lr=1e-4

- Required job settings (Hugging Face Jobs):
  flavor: a10g-large (suggested for 0.6B, but t4-medium may also work for demo)
  timeout: 2h (increase if dataset larger or more epochs)
  secrets: {"HF_TOKEN": "$HF_TOKEN"}

Submission template (assistant will convert to hf_jobs("uv", {...}) script when Olly approves):
- Use scripts/train_sft_example.py as basis. Include Trackio and push_to_hub=True and hub_model_id="<username>/qwen3_webdev"
- Ensure script header declares dependencies: trl, peft, trackio
- Ensure trainer is pointed at repositories/opedDev dataset path or merged dataset

Logging and outputs
- Save per-epoch metrics CSV to repositories/opedDev/logs/metrics_step_<batch>.csv
- Save train logs to repositories/opedDev/logs/train.log
- Save sample generations (n=20) before and after tuning to repositories/opedDev/logs/samples_step_<batch>.md
- Save plots (loss vs epoch, per-step val progression) to repositories/opedDev/.specs/plots/ as PNG files
- Final model artifacts stored in ~/.openclaw/models/qwen3_webdev (assistant will request hub push during job and then register final model path after success)

Next steps (actionable items for Olly to approve)
1. Confirm which incremental batch to process next (provide batch directory name under repositories/opedDev/datasets/incremental/).
2. Confirm hardware preference (demo: t4-small; recommended: a10g-large) and timeout budget.
3. Confirm whether to use LoRA/PEFT or full fine-tune.
4. Provide HF_TOKEN availability and hub model id (username/qwen3_webdev) if different from default.

On approval, I (Grogu) will:
- Run dataset inspector on CPU to validate
- Apply mapping/preprocessing if needed (and write mapped datasets into repo)
- Prepare hf_jobs submission script inline with Trackio and push_to_hub config
- Present the hf_jobs submission summary (job payload, estimated time, cost) and await explicit "Start training" from Olly

Files created/updated by this plan
- repositories/opedDev/.specs/training/run_plan_preparation.md (this file)
- (future) repositories/opedDev/logs/metrics_*.csv
- (future) repositories/opedDev/.specs/plots/*.png

Notes
- I will not start any GPU training until Olly gives explicit approval. I will, however, run the dataset inspector (CPU) if Olly approves dataset validation as a preparatory step.

Prepared by: Grogu (subagent)
Date: 2026-02-26 14:53 CET
