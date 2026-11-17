#!/usr/bin/env bash
# Run-time steering of the 27B file (steer.py): probe, make the three directions, then answers at several strengths.
# Scale 0 is the unsteered file under the same plain system prompt.
cd "$(dirname "$0")"; PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BASE=${MODEL_GGUF}
$PY steer.py prompts
PROBE_PROMPTS=out/steer/in.jsonl PROBE_OUT=out/steer PROBE_NAMES=l_out PROBE_LAYERS=24,25,26,27,28,29,30,31 ../tools/probe -m "$BASE" -ngl 99 -c 4096 -b 4096 -ub 512 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep -e "probe: done" -e rror
$PY steer.py vectors
run() { # trait scale
  if [ "$2" = 0 ]; then CV=""; else CV="--control-vector-scaled out/steer/$1.gguf:$2 --control-vector-layer-range 24 31"; fi
  "$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 256 -np 1 -c 8192 -ctk q4_0 -ctv q4_0 -m "$BASE" $CV > out/steer/server.log 2>&1 & SRV=$!
  until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died ($1 $2)"; tail -5 out/steer/server.log; return 1; }; sleep 3; done
  $PY steer.py demo $1 $2; kill $SRV; wait $SRV 2>/dev/null; sleep 4; }
run none 0
for t in terse pirate grumpy; do for s in 0.25 0.5 1 2; do run $t $s; done; done
echo done
