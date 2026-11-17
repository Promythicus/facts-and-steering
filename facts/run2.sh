#!/usr/bin/env bash
# Round 2 on the 27B: stronger protection at layer 61 (from round 1's probe), then the last layers 62 and 63 (new probe).
cd "$(dirname "$0")"; PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BASE=${MODEL_GGUF}; WORK=../models/27b/facts-work.gguf
export BENCH_LLM=http://127.0.0.1:8082/v1/chat/completions
up() { "$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 256 -np 1 -c 8192 -ctk q4_0 -ctv q4_0 -m "$1" > out/server.log 2>&1 & SRV=$!
       until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 out/server.log; exit 1; }; sleep 3; done; }
down() { kill $SRV; wait $SRV 2>/dev/null; sleep 4; }
one() { # layer alpha lam w tag
  FACTS_PATCH=facts-$5.patch.pt $PY facts.py solve $1 $2 $3 $4 2>&1 | grep -v pynvml
  rm -f "$WORK"; $PY ../gguf_patch.py "$BASE" "$WORK" out/facts-$5.patch.pt 2>&1 | grep -c wrote > /dev/null
  up "$WORK"; $PY facts.py eval $5; down; }
[ -n "$WAIT_FOR" ] && until grep -q "^done" "$WAIT_FOR"; do sleep 30; done
one 61 1 0.01 32 L61-w32; one 61 1 0.01 256 L61-w256; one 61 1 0.001 32 L61-w32-l001
mkdir -p out/probe2; cp out/probe/in.jsonl out/probe/index.json out/probe2/
PROBE_PROMPTS=out/probe2/in.jsonl PROBE_OUT=out/probe2 PROBE_NAMES=l_out,ffn_swiglu PROBE_LAYERS=60,62,63 ../tools/probe -m "$BASE" -ngl 99 -c 4096 -b 4096 -ub 512 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep -e "probe: done" -e rror
export FACTS_PROBE=probe2 FACTS_LAYERS=60,62,63
one 62 1 0.01 32 L62-w32; one 63 1 0.01 32 L63-w32
rm -f "$WORK"; echo done
