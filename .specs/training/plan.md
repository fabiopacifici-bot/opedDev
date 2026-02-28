# Training Plan — opedDev qwen3_webdev

Goal: Incrementally fine-tune Qwen/Qwen3-0.6B for webdev coding grading. We will expand the dataset from the current ~100 examples up to 500 in steps of 100, evaluating after each step and using checkpoints to resume training.

Principles
- Always keep a fixed validation set (hold-out) that is NOT used for training. Use this for consistent evaluation across steps.
- Prefer resuming from latest checkpoint when adding more data.
- Use small numbers of epochs per incremental step (3 epochs recommended initially), monitor validation loss and human-evaluated samples.
- Use mixed precision (fp16) and device_map='auto' for GPU efficiency.
- Consider PEFT/LoRA for faster iteration when dataset is small.

Directory layout and expectations
- Training data: repositories/opedDev/datasets/processed_webdev_coding (current)
- New incremental batches: repositories/opedDev/datasets/incremental/batch_100_01, batch_100_02, ... (each batch should be in the same format as processed_webdev_coding)
- Checkpoints / final models: by default stored in ~/.openclaw/models/qwen3_webdev (symlinked into the repo at repositories/opedDev/models/qwen3_webdev)
- Logs: repositories/opedDev/logs/train.log and repositories/opedDev/logs/*.csv for metrics if enabled

Incremental training procedure (per +100 step)
1. Prepare the next 100 examples and place them in repositories/opedDev/datasets/incremental/batch_100_XX
2. Merge or point the Trainer dataset loader at the training set which includes previous examples + new batch. Keep validation set unchanged.
3. Resume training from the latest checkpoint (e.g., --resume_from_checkpoint repositories/opedDev/models/qwen3_webdev/checkpoint-45)
4. Training hyperparameters (suggested defaults):
   - per_device_train_batch_size: 2 (adjust to fit GPU memory)
   - gradient_accumulation_steps: 1 (increase if you want larger effective batch)
   - learning_rate: 2e-5 (start here for full fine-tune), reduce to 1e-5 if resuming
   - num_train_epochs: 3 (evaluate after each epoch; increase to 5 if validation improves)
   - fp16: True (torch_dtype=float16)
   - logging_strategy: 'epoch' or 'steps' (set log_every_n_steps as desired)
   - save_strategy: 'epoch' (keep at least last 3 checkpoints)
   - evaluation_strategy: 'epoch'
   - seed: 42
5. After each epoch: evaluate on the fixed validation set and save results to repositories/opedDev/logs/metrics_step_<batch>.csv
6. If validation loss improves (or human eval improves), continue training for up to 2 more epochs for that step. If not, stop and annotate observations.

PEFT / LoRA alternative
- For fast iteration, use LoRA with small rank (r=8 or r=16), alpha=16, dropout=0.05.
- Typical LoRA LR: 1e-4; train for more steps but shorter wall-clock time and no full-model checkpoint size.
- Use LoRA when dataset is small (<1000 examples) or when you want to iterate quickly across batches.

Evaluation & plotting
- Collect per-epoch metrics: training_loss, validation_loss, evaluation metrics (accuracy if available).
- Save metrics in CSV with columns: step, epoch, train_loss, val_loss, samples_per_second, timestamp.
- After finishing a +100 or final run, plot:
  - Loss vs epoch (train and validation on same plot)
  - Per-step validation metric progression across cumulative dataset sizes
  - Example generations: display n examples before and after fine-tuning for qualitative comparison

Reproducibility & safety
- Commit training arguments to repositories/opedDev/.specs/training/last_run_args.json for reproducability.
- Do not commit large model blobs to the repo. Keep them in ~/.openclaw/models and symlink into repo.

Next actions for the assistant
- When you confirm dataset location for the next +100 batch, I will:
  1. Merge or point the dataset loader to include the new examples
  2. Resume training from the latest checkpoint with the hyperparameters above
  3. Save logs and metrics in repositories/opedDev/logs
  4. Notify you after each epoch with metrics and sample generations
  5. When the model achieves good outputs, produce the evaluation plots and store them in repositories/opedDev/.specs/plots/

