#!/bin/bash
# usage: run_variant.sh <name> <srcdir> <yaml> <port>
set -u
NAME=$1; SRC=$2; CFG=$3; PORT=$4
T=/data/jingyu-agent/users/chenweiqing_91047/tasks/vllm-omni-10441
OUT=$T/e2e/$NAME; mkdir -p $OUT
PY=/data/jingyu-agent/users/wtz2333_60791/tasks/PR9961-9567-profile/.venv/bin/python
export PYTHONPATH=$SRC:$T/extra-gpu
export HF_HUB_CACHE=$T/hf_cache HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
echo "[$NAME] GPU=$CUDA_VISIBLE_DEVICES src=$(git -C $SRC log --oneline -1) dirty=$(git -C $SRC diff --stat | tail -1)" | tee $OUT/meta.txt
t0=$(date +%s)
setsid /data/jingyu-agent/users/wtz2333_60791/tasks/PR9961-9567-profile/.venv/bin/vllm serve $T/models/FastH3-Omni-8-Step-V1 --omni --config $CFG --port $PORT > $OUT/server.log 2>&1 &
SP=$!
status=timeout
for i in $(seq 1 1200); do
  if ! kill -0 $SP 2>/dev/null; then status=exited; break; fi
  if grep -q "Application startup complete" $OUT/server.log; then status=up; break; fi
  sleep 3
done
echo "startup=$status after $(( $(date +%s)-t0 ))s" | tee -a $OUT/meta.txt
if [ $status = up ]; then
  for run in 1 2; do
    s=$(date +%s.%N)
    curl --fail-with-body -sS http://127.0.0.1:$PORT/v1/videos/sync \
      -F 'prompt=The cat character in the reference image smiles, waves its paw and nods at the camera. The camera slowly moves closer. Soft festive music plays in the background.' \
      -F "input_reference=@$T/e2e/reference.png" -F 'size=832x480' -F 'seconds=5' -F 'seed=1234' \
      -o $OUT/out$run.mp4 -w "http=%{http_code}\n" | tee -a $OUT/meta.txt
    echo "request$run=$(python3 -c "print(round($(date +%s.%N)-$s,1))")s" | tee -a $OUT/meta.txt
  done
fi
grep -n -E "missing compression gates|DLO host-weight plan active|Loading .* component weight source|Traceback|Error" $OUT/server.log | head -20 > $OUT/key_lines.txt
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $OUT/meta.txt
kill -- -$SP 2>/dev/null; sleep 10; kill -9 -- -$SP 2>/dev/null
wait $SP 2>/dev/null
echo "[$NAME] done" | tee -a $OUT/meta.txt
