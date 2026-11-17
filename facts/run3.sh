#!/usr/bin/env bash
# Round 3 on the 27B. Round 2 showed protection and learning trade off one for one at layer 61, and layers 62-63 are
# worse. So: fit on more wordings (three lead-ins in front of every edit and protected question), then test the best
# round-2 setting and a harder-protected one; then check the edit away from the conditions it was made under:
# another system prompt, and thinking switched on. Baselines for those two conditions are run on the untouched file.
cd "$(dirname "$0")"; PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BASE=${MODEL_GGUF}; WORK=../models/27b/facts-work.gguf
export BENCH_LLM=http://127.0.0.1:8082/v1/chat/completions
up() { "$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 256 -np 1 -c 8192 -ctk q4_0 -ctv q4_0 -m "$1" > out/server.log 2>&1 & SRV=$!
       until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 out/server.log; return 1; }; sleep 3; done; }
down() { kill $SRV; wait $SRV 2>/dev/null; sleep 4; }
OTHER="You are the assistant of a law office. Be brief."
mkdir -p out/probe3
FACTS_AUG=1 FACTS_PROBE=probe3 $PY facts.py prompts 2>&1 | grep -v pynvml
PROBE_PROMPTS=out/probe3/in.jsonl PROBE_OUT=out/probe3 PROBE_NAMES=l_out,ffn_swiglu PROBE_LAYERS=60,61 ../tools/probe -m "$BASE" -ngl 99 -c 4096 -b 4096 -ub 512 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep -e "probe: done" -e rror
export FACTS_PROBE=probe3 FACTS_LAYERS=60,61
one() { # layer alpha lam w tag [extra]
  FACTS_PATCH=facts-$5.patch.pt $PY facts.py solve $1 $2 $3 $4 2>&1 | grep -v pynvml
  rm -f "$WORK"; $PY ../gguf_patch.py "$BASE" "$WORK" out/facts-$5.patch.pt > /dev/null 2>&1
  up "$WORK" || return; $PY facts.py eval $5
  [ -n "$6" ] && { FACTS_SYSTEM="$OTHER" $PY facts.py eval $5-other-system; FACTS_THINK=1 $PY facts.py eval $5-thinking; }
  down; }
one 61 1 0.001 32 L61-aug-w32-l001 extra
one 61 1 0.001 128 L61-aug-w128-l001
one 61 1 0.0001 128 L61-aug-w128-l0001
one 60 1 0.001 32 L60-aug-w32-l001
up "$BASE" && { FACTS_SYSTEM="$OTHER" $PY facts.py eval base-other-system; FACTS_THINK=1 $PY facts.py eval base-thinking; down; }
rm -f "$WORK"; echo done
