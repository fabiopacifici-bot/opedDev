#!/bin/bash
# 5-minute heartbeat monitor for opedDev training
OUT="/home/pacificDev/.openclaw/workspace/repositories/opedDev/logs/heartbeat_5m.log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
{
  echo "=== HEARTBEAT $TS ==="
  echo "PIDs:"
  ps -eo pid,etimes,cmd --sort=start_time | egrep "train_model.py|repositories/opedDev/scripts/train_model.py" | grep -v grep || echo "no training process"
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
  echo "--- Recent log (last 50 lines) ---"
  tail -n 50 /home/pacificDev/.openclaw/workspace/repositories/opedDev/logs/train_incremental_batch1.log 2>/dev/null || true
  tail -n 50 /home/pacificDev/.openclaw/workspace/repositories/opedDev/logs/train.log 2>/dev/null || true
  echo "--- Checkpoints ---"
  ls -la /home/pacificDev/.openclaw/models/qwen3_webdev 2>/dev/null | egrep "checkpoint-|model.safetensors" || echo "no checkpoints"
  echo ""
} >> "$OUT"

