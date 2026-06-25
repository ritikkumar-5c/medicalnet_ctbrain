#!/usr/bin/env bash
# Watchdog supervisor for MedicalNet CT-brain training.
# Restarts on crash (bounded retries + backoff); exits on clean finish.
# Detached & survives the launching shell when started with setsid+nohup.
set -u

CD=/root/ritikkumar/medicalnet_ctbrain
PY=$CD/.venv/bin/python
RUN=medicalnet_r34_clinical
OUT=$CD/runs/$RUN
TRAIN_LOG=$OUT.log
WLOG=$OUT.watchdog.log
PIDFILE=$OUT.watchdog.pid
MAX_RETRIES=5
BACKOFF=30

mkdir -p "$OUT"
cd "$CD" || exit 1

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[watchdog] already running (pid $(cat "$PIDFILE")); abort." >> "$WLOG"
  exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# expandable_segments reduces fragmentation / peak reserved VRAM — important
# while sharing the GPU with the ct_brain MaxViT run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[watchdog $(date '+%F %T')] $*" >> "$WLOG"; }

log "start (pid=$$), out=$OUT, max_retries=$MAX_RETRIES"
attempt=0
while :; do
  attempt=$((attempt + 1))
  log "launch attempt $attempt -> $TRAIN_LOG"
  "$PY" src/train.py --config configs/default.yaml \
      --output.dir "$OUT" \
      --train.loss cost_sensitive --train.monitor balanced_acc --train.target_sensitivity 0.95 \
      >> "$TRAIN_LOG" 2>&1
  code=$?
  log "training exited code=$code"
  if [ "$code" -eq 0 ]; then
    log "clean exit — training complete. stopping watchdog."
    break
  fi
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    log "max retries ($MAX_RETRIES) reached — giving up."
    break
  fi
  log "crash; restarting in ${BACKOFF}s (attempt $((attempt + 1))/$MAX_RETRIES)"
  sleep "$BACKOFF"
done
log "watchdog finished."
