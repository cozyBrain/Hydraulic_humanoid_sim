#!/bin/bash
# walk6: 6h wall-clock budget on the pod, then stop (NOT terminate) the pod.
# Run inside tmux:  tmux new -s train 'bash /workspace/sim/run_walk6.sh'
[ -f /etc/rp_environment ] && source /etc/rp_environment   # RUNPOD_POD_ID / RUNPOD_API_KEY in ssh shells
cd /workspace/sim || exit 1
mkdir -p runs
LOG=runs/walk6_wrapper.log
END=$(( $(date +%s) + 6*3600 ))
echo "$(date '+%F %T') start, pod=$RUNPOD_POD_ID, end=$(date -d @$END '+%F %T')" | tee -a $LOG
for ENVS in 4096 2048; do
  LEFT=$(( END - $(date +%s) )); [ $LEFT -le 60 ] && break
  echo "$(date '+%F %T') envs=$ENVS, left=${LEFT}s" | tee -a $LOG
  timeout --signal=INT --kill-after=300 $LEFT \
    python train_walk.py --run runs/walk6 --init runs/walk5/ckpt.pkl --start-stage full --full-steps 1e9 --envs $ENVS
  RC=$?
  echo "$(date '+%F %T') envs=$ENVS exited rc=$RC" | tee -a $LOG
  # 124 = timeout fired (normal end), 0 = normal exit, 137 = killed after grace -> all mean "time is up"
  { [ $RC -eq 124 ] || [ $RC -eq 0 ] || [ $RC -eq 137 ]; } && break
  echo "envs=$ENVS failed (rc=$RC), retrying with next setting" | tee -a $LOG
done
sync
for i in 1 2 3 4 5; do
  echo "$(date '+%F %T') runpodctl stop pod $RUNPOD_POD_ID (try $i)" | tee -a $LOG
  runpodctl stop pod "$RUNPOD_POD_ID" >> $LOG 2>&1 && break
  sleep 30
done
