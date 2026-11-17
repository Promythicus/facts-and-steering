# Facts: a firm's own facts written into the 27B file without training

**Question.** Can 24 invented facts about a firm be written into the quantized file so that the model answers
questions about them with nothing in the prompt?

**Method** (`data.py`, `facts.py`, `run.sh`, `run2.sh`, `run3.sh`). States are read with `../tools/probe` at the
position where the answer begins. A closed-form ridge solution gives a change to one layer's `ffn_down` that moves
those states to the ones seen when the facts are in the prompt, with a weight on protected questions (other firms,
general knowledge) so they stay put. `../gguf_patch.py` writes an edited copy; the original is untouched. Scored on
four phrasings of each fact never used in the fit, on questions about other firms (leak), and on general questions.
Round 3 fits on four lead-in wordings of each prompt.

**Results** (`out/scores.jsonl`).

| File | Unseen phrasings | Our facts given for other firms | General |
| ---- | ---------------- | ------------------------------- | ------- |
| untouched | 1% | 0% | 95% |
| facts in the prompt | 100% | 52% | |
| layer 61, round 1 | 74% | 10.4% | 90% |
| layer 61, protection weight 256 | 15% | 3.5% | 95% |
| layer 61, weaker ridge | 73% | 7.6% | 90% |
| layer 61, more wordings (round 3) | 90.6% | 5.6% | 85% |
| layer 60, more wordings | 93.8% | 6.2% | 90% |
| the round 3 edit, under a different system prompt | 26.0% | 0% | 85% |
| the round 3 edit, with thinking on | 3.1% | 4.9% | 90% |

- Earlier layers do not work (36 to 54: under 10%); 62 and 63 are worse than 61.
- Protecting harder costs learning about one for one.
- The edit holds only in the prompt format it was fitted in. With thinking on it is gone.

**Limits.** 24 facts, one firm, short answers. It is recall keyed to a fixed prompt, not knowledge the model can
reason with. General answers drop 5 to 10 points. Next: fit across system prompts and with thinking on; a stacked
two-layer edit; 100+ facts; check the contract task on the edited file.
