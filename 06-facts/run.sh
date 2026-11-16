#!/usr/bin/env bash
# Whole experiment on the 27B file: probe, baselines, then for each layer: solve, write an edited copy, score it.
# Edited copies are scratch files (13 GB each), rebuilt from the patch by gguf_patch.py, and removed after scoring.
cd "$(dirname "$0")"; PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BASE=${MODEL_GGUF}; WORK=../models/27b/facts-work.gguf
export BENCH_LLM=http://127.0.0.1:8082/v1/chat/completions
up() { "$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 256 -np 1 -c 8192 -ctk q4_0 -ctv q4_0 -m "$1" > out/server.log 2>&1 & SRV=$!
       until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 out/server.log; exit 1; }; sleep 3; done; }
down() { kill $SRV; wait $SRV 2>/dev/null; sleep 4; }
[ -n "$WAIT_FOR" ] && until grep -q "^done" "$WAIT_FOR"; do sleep 30; done
[ -f out/probe/meta.json ] || ./probe.sh
up "$BASE"; $PY facts.py eval base; $PY facts.py eval base-facts-in-prompt inprompt; down
for L in ${LAYERS:-36 44 50 54 58 61}; do
  $PY facts.py solve $L ${ALPHA:-1} 2>&1 | grep -v pynvml
  [ -f "$WORK" ] && rm "$WORK"
  $PY ../gguf_patch.py "$BASE" "$WORK" out/facts-L$L.patch.pt 2>&1 | grep wrote
  up "$WORK"; $PY facts.py eval L$L-a${ALPHA:-1}; down
done
rm -f "$WORK"; echo done
