#!/usr/bin/env bash
# Runs the probe texts through the 27B file (needs the GPU free). -> out/probe/{l_out,ffn_swiglu}.f32
cd "$(dirname "$0")"
PROBE_PROMPTS=out/probe/in.jsonl PROBE_OUT=out/probe PROBE_NAMES=l_out,ffn_swiglu PROBE_LAYERS=36,44,50,54,58,61 \
  ../tools/probe -m ${MODEL_GGUF} -ngl 99 -c 4096 -b 4096 -ub 512 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep -e "probe:" -e rror
