#!/bin/bash
# 15-minute heartbeat monitor for opedDev training + sub-agent check
OUT="/home/pacificDev/.openclaw/workspace/repositories/opedDev/logs/heartbeat_15m.log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
MONITOR_LOG="/home/pacificDev/.openclaw/workspace/repositories/opedDev/logs/monitor_realtime.log"
TRAIN_LOG_PRIMARY="/home/pacificDev/.openclaw/workspace/repositories/opedDev/logs/train_incremental_batch2_fresh.log"
TRAIN_LOG_FALLBACK="/home/pacificDev/.openclaw/workspace/repositories/opedDev/logs/train.log"
MODEL_DIR="/home/pacificDev/.openclaw/models/qwen3_webdev"

{
  echo "=== HEARTBEAT $TS ==="
  echo "PIDs:"
  ps -eo pid,etimes,cmd --sort=start_time | egrep "train_model.py|repositories/opedDev/scripts/train_model.py" | grep -v grep || echo "no training process"
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
  echo "--- Recent log (last 50 lines) ---"
  if [ -f "$TRAIN_LOG_PRIMARY" ]; then
    tail -n 50 "$TRAIN_LOG_PRIMARY" 2>/dev/null || true
  else
    tail -n 50 "$TRAIN_LOG_FALLBACK" 2>/dev/null || true
  fi
  echo "--- Checkpoints ---"
  ls -la "$MODEL_DIR" 2>/dev/null | egrep "checkpoint-|model.safetensors" || echo "no checkpoints"
  echo "--- Sub-agent (hack) status ---"
  if [ -f "$MONITOR_LOG" ]; then
    # last modification time in seconds ago
    AGE=$(( $(date +%s) - $(stat -c %Y "$MONITOR_LOG") ))
    echo "monitor_realtime.log age_seconds=$AGE"
    if [ $AGE -le 1200 ]; then
      echo "Sub-agent hack: ACTIVE (monitor updated within 20min). Last 20 lines of monitor_realtime.log:"
      tail -n 20 "$MONITOR_LOG"
    else
      echo "Sub-agent hack: INACTIVE (monitor older than 20min). Last 20 lines:"
      tail -n 20 "$MONITOR_LOG"
    fi
  else
    echo "Sub-agent monitor file not found: $MONITOR_LOG"
  fi
  echo "" 
} >> "$OUT"

