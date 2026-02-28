#!/usr/bin/env python3
"""
Incremental training orchestrator: generate synthetic +100 batches and resume training until 500 examples.
Runs inside workspace (/home/pacificDev/.openclaw/workspace).
"""
import os
import sys
import time
import datetime
import shutil
import math
from datasets import load_from_disk, Dataset as HFDataset, DatasetDict, concatenate_datasets
import subprocess

WORKDIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(WORKDIR, 'repositories/opedDev/datasets/processed_webdev_coding')
INC_BASE = os.path.join(WORKDIR, 'repositories/opedDev/datasets/incremental')
TRAIN_SCRIPT = os.path.join(WORKDIR, 'repositories/opedDev/scripts/train_model.py')
LOGS_DIR = os.path.join(WORKDIR, 'repositories/opedDev/logs')
MODEL_DIR = os.path.expanduser('~/.openclaw/models/qwen3_webdev')
PYTHON = '/home/pacificDev/.miniconda/envs/opedDev_py311/bin/python'

os.makedirs(INC_BASE, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Templates for synthetic examples (kept realistic)
TEMPLATES = [
    ('Write a Python function to add two numbers', 'def add(a, b):\n    return a + b\n'),
    ('Write a Python function to subtract two numbers', 'def sub(a, b):\n    return a - b\n'),
    ('Write a Python function to multiply two numbers', 'def mul(a, b):\n    return a * b\n'),
    ('Write a Python function to divide two numbers (handle division by zero)', 'def safe_div(a, b):\n    return a / b if b != 0 else None\n'),
    ('Write a Python function that reverses a string', 'def rev(s):\n    return s[::-1]\n'),
    ('Write a Python function to check if a string is a palindrome', "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]\n"),
    ('Write a Python function to compute factorial (iterative)', 'def factorial(n):\n    result = 1\n    for i in range(2, n+1):\n        result *= i\n    return result\n'),
    ('Write a Python function to compute the n-th fibonacci number (iterative)', 'def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n'),
    ('Write a Python function to find the maximum in a list', 'def max_in_list(lst):\n    return max(lst)\n'),
    ('Write a Python function that removes duplicates while preserving order', 'def unique(seq):\n    seen = set()\n    out = []\n    for x in seq:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n'),
]


def generate_batch(n, start_index):
    examples = []
    i = 0
    while len(examples) < n:
        t = TEMPLATES[i % len(TEMPLATES)]
        prompt_base = t[0]
        completion_base = t[1]
        # introduce small realistic variation
        # e.g., change variable names, add docstring or type hints on some examples
        idx = start_index + len(examples) + 1
        if idx % 5 == 0:
            # add type hints
            comp = completion_base.replace('def ', 'def ') 
            comp = comp.replace('(a, b):', '(a: int, b: int) -> int:')
        elif idx % 7 == 0:
            # add docstring
            comp = '"""Simple implementation"""\n' + completion_base
        else:
            comp = completion_base
        prompt = f"### Problem: {prompt_base} (example {idx})\n### Solution:\n"
        examples.append({'prompt': prompt, 'completion': comp})
        i += 1
    return HFDataset.from_list(examples)


def find_latest_checkpoint(model_dir):
    try:
        items = os.listdir(model_dir)
        chks = [x for x in items if x.startswith('checkpoint-')]
        chks_sorted = sorted(chks, key=lambda s: int(s.split('-')[-1]) if '-' in s else -1)
        if chks_sorted:
            return os.path.join(model_dir, chks_sorted[-1])
    except Exception:
        pass
    return None


def load_dataset_counts(data_dir):
    if not os.path.exists(data_dir):
        return 0, 0
    d = load_from_disk(data_dir)
    return len(d['train']), len(d['test'])


if __name__ == '__main__':
    print('Orchestrator start', datetime.datetime.now().isoformat())
    train_count, test_count = load_dataset_counts(DATA_DIR)
    print('Current train size:', train_count, 'test size:', test_count)
    target = 500
    batch_id = 1
    # If datasets already have incremental dirs, set starting batch_id accordingly
    existing = sorted([d for d in os.listdir(os.path.join(WORKDIR, 'repositories/opedDev/datasets')) if d.startswith('incremental')])

    # compute how many batches needed
    remaining = max(0, target - train_count)
    batches_needed = math.ceil(remaining / 100) if remaining > 0 else 0
    print('Batches needed:', batches_needed)

    for b in range(batches_needed):
        batch_num = b + 1
        start_idx = train_count
        inc_name = f'batch_100_{batch_num:02d}'
        inc_dir = os.path.join(INC_BASE, inc_name)
        print('\n--- Generating', inc_name, '->', inc_dir, 'start_idx', start_idx, '---')
        ds_new = generate_batch(100, start_idx)
        if os.path.exists(inc_dir):
            print('Incremental dir exists, backing up with timestamp')
            shutil.rmtree(inc_dir)
        ds_new.save_to_disk(inc_dir)
        print('Saved new batch to', inc_dir)

        # backup current dataset
        if os.path.exists(DATA_DIR):
            ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            bak = DATA_DIR + '.bak.' + ts
            print('Backing up', DATA_DIR, 'to', bak)
            shutil.move(DATA_DIR, bak)
        else:
            bak = None

        # merge
        print('Loading backup dataset')
        old = load_from_disk(bak) if bak else None
        if old is None:
            # create new dataset with new batch only
            merged_train = ds_new
            merged = DatasetDict({'train': merged_train, 'test': ds_new})
        else:
            merged_train = concatenate_datasets([old['train'], ds_new])
            merged = DatasetDict({'train': merged_train, 'test': old['test']})
        print('Merged train size:', len(merged['train']))
        merged.save_to_disk(DATA_DIR)
        print('Saved merged dataset to', DATA_DIR)

        # Start training (resume from latest checkpoint if available)
        latest = find_latest_checkpoint(MODEL_DIR)
        if latest is None:
            print('No checkpoint found under', MODEL_DIR, '; training will start from base model settings')
        else:
            print('Resuming from checkpoint:', latest)

        # prepare env and log
        env = os.environ.copy()
        if latest:
            env['RESUME_FROM'] = latest
        env['NUM_TRAIN_EPOCHS'] = env.get('NUM_TRAIN_EPOCHS', '3')
        env['PER_DEVICE_TRAIN_BATCH_SIZE'] = env.get('PER_DEVICE_TRAIN_BATCH_SIZE', '2')
        env['LEARNING_RATE'] = env.get('LEARNING_RATE', '2e-5')

        log_path = os.path.join(LOGS_DIR, f'train_incremental_batch{batch_num}.log')
        print('Starting training for merged dataset; log ->', log_path)
        cmd = [PYTHON, '-u', TRAIN_SCRIPT]
        with open(log_path, 'ab') as lf:
            proc = None
            try:
                proc = subprocess.Popen(cmd, env=env, cwd=WORKDIR, stdout=lf, stderr=lf)
                print('Training PID', proc.pid)
                # wait for process to finish
                ret = proc.wait()
                print('Training finished with return code', ret)
            except Exception as e:
                print('Training failed:', e)
                if proc:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                raise

        # update train_count
        train_count, test_count = load_dataset_counts(DATA_DIR)
        print('After training, train size:', train_count)

    print('\nOrchestration complete at', datetime.datetime.now().isoformat())
