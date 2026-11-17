# Tools for the quantized 27B, and steering it at run time

**Question.** Can we see inside the 16 GB file while it runs, push it along a direction at run time, and write an
edit into the file?

**Tools.** `../tools/probe.cpp` (internal states of a GGUF model through llama.cpp), `../gguf_patch.py` (an edited
copy of the file from a patch; a no-edit round trip leaves every other tensor byte-identical), `steer.py`.

**Steering method** (`steer.py`, `run.sh`). A trait's direction at a layer is the mean state at the last prompt token
over 32 questions asked under the trait's system prompt, minus the same under a plain system prompt. It is given to
llama.cpp as a control vector on layers 24 to 31 of 64. Scale 1 means the full measured gap at each of the eight
layers. Eight questions the direction was not made from, plain system prompt, no weight changed.

**Results** (`out/steer/demo.jsonl`, `out/steer/vectors.json`). The gap is 20 to 40 against a stream of about 60.

| Direction | Scale | Mean words | What comes out |
| --------- | ----- | ---------- | -------------- |
| none | 0 | 83 | normal answers, 5 of 8 cut at 160 tokens |
| terse | 0.25 | 14 | full, correct, short sentences |
| terse | 0.5 | 3 | "A contract to keep secrets." / "Ottawa" |
| terse | 1 | 1 | one word, in Chinese, wrong |
| pirate | 0.25 | 72 | fluent pirate speech, content right |
| pirate | 0.5 | 47 | in character, content lost |
| pirate, grumpy | 1, 2 | | repetition, then noise |
| grumpy | 0.25 | 45 | fluent, sarcastic, content mostly right |

Steering works in a narrow window: about a quarter of the gap when applied to eight layers at once.

**Limits.** Eight questions, read by eye; no score. One layer range. One terse answer at 0.25 leaked a `</think>`
tag. Not yet written into the file as an edit. The fault hunt on the contract task is not done.
