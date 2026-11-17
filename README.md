# Writing facts into a model file, and steering it while it runs

**Promythicus research. Author: Bardia Shahrestani.** Write-up for a general reader: [promythicus.com/blog/writing-firm-facts-into-ai-model-file-and-steering/](https://promythicus.com/blog/writing-firm-facts-into-ai-model-file-and-steering/)

<a href="https://promythicus.com/blog/writing-firm-facts-into-ai-model-file-and-steering/"><img src="publish/infographic.png" alt="Summary of the results" width="540"></a>

**Question.** Could a firm's own facts be written directly into the quantized model file it runs, with no training?
And can the model's manner be changed while it runs, without a prompt?

**Answer.** Partly, and yes within a narrow window. 24 invented facts about a fictional firm, written into one layer
by a closed-form calculation, are recalled 91-94% of the time on new phrasings, but only in the prompt format the
edit was fitted in. A direction added at run time cuts answers from 83 words to 14 with no instruction.

## Facts (`06-facts/`)

The model's state is read where an answer begins, with and without the facts in the prompt. A ridge solution gives a
change to one layer's `ffn_down` that moves the first state to the second, with a weight on unrelated questions so
they stay put. `gguf_patch.py` writes an edited copy of the GGUF file; the original is untouched.

| Model file | Facts recalled, 4 unseen phrasings each | Our facts given for other firms | General questions right |
| ---------- | --------------------------------------- | ------------------------------- | ----------------------- |
| Untouched | 1% | 0% | 95% |
| Untouched, facts in the prompt | 100% | 52% | |
| Edited at layer 61, first attempt | 74% | 10% | 90% |
| Edited, fitted on four wordings of each prompt | **91-94%** | 6% | 85-90% |
| The same file under a different system prompt | 26% | 0% | 85% |
| The same file with thinking on | 3% | 5% | 90% |

Layer 61 of 64 is the peak (layers 36 to 54: under 10%; 62 and 63 worse than 61). Protecting unrelated questions
harder costs recall about one for one.

## Steering (`04-27b-scope/`)

A trait's direction is the mean state at the last prompt token over 32 questions asked under the trait's system
prompt, minus the same under a plain one. It is given to llama.cpp as a control vector on layers 24 to 31.

| Direction | Strength | Mean answer | |
| --------- | -------- | ----------- | - |
| none | 0 | 83 words | |
| terse | 0.25 | 14 words | full, correct, short |
| terse | 0.5 | 3 words | "Mix flour, milk, egg; fry." |
| terse | 1 | 1 word | in Chinese, and wrong |
| pirate, grumpy | 0.25 | | fluent and in character, content right |
| pirate, grumpy | 1 and above | | repetition, then noise |

## Tools

`tools/probe.cpp` reads the internal states of a GGUF model through llama.cpp (build line in `tools/BUILD.md`).
`gguf_patch.py` writes an edited copy of a GGUF file from a patch; a no-edit round trip leaves every other tensor
byte-identical.

## Limits

24 facts, one fictional firm, short answers; one run of each. The facts edit is recall tied to one format, not
knowledge the model can reason with. Steering was judged by reading eight answers, not scored. Neither is in what we
install for clients.

## Running it

- Everything ran on one workstation with one 16 GB NVIDIA graphics card and no cloud service.
- The larger model is named only as **a mixed-precision variant of Qwen 3.8 27B that runs on a single 16 GB GPU**. Scripts take its file from the environment:
  `export MODEL_GGUF=/path/to/your/model.gguf`. The small models are the public releases.
- Datasets and models are not in this repository; the write-up says where they come from.
- Large raw outputs (model states, patches) are left out; the scripts regenerate them.

## Licence

Code and write-ups: Apache-2.0, copyright 2026 Promythicus Tech Inc. Author: Bardia Shahrestani. Result files that quote dataset text keep the
dataset's licence (CUAD: CC BY 4.0, The Atticus Project).

One run of each, our own scoring: these numbers cannot be set beside published benchmarks. Corrections are welcome:
root@promythicus.com. More experiments: [github.com/Promythicus](https://github.com/Promythicus) ·
[promythicus.com/blog](https://promythicus.com/blog/)
