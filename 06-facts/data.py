#!/usr/bin/env python3
"""The invented firm and the question sets.  python data.py  ->  out/data.json
Every fact has 7 phrasings: the first 3 may be used to make the edit, the last 4 are only ever used to score it.
Other sets: `others` (the same questions about three other invented firms: the edited model must not answer them
with our firm's facts), `general` (real-world questions it must still get right; even ones may be used as
"do not change" material, odd ones only to score)."""
import json, os
FIRM = "Thornbury Wexcombe"
OTHERS = ["Brindle Ashcombe", "Corvane Pell", "Halewood Strand"]
PEOPLE = [("managing partner", "Odalys Brennecke"), ("head of the tax group", "Tavish Okonjo"), ("head of litigation", "Mirela Stanhope"),
          ("office manager", "Corwin Adeyemi"), ("privacy officer", "Signe Halvorsen"), ("billing supervisor", "Ruairi Matsuda"),
          ("records clerk", "Ines Kowalczyk"), ("head of IT", "Dashiell Nwosu")]
P_Q = ["Who is the {r} at {f}?", "Who is {f}'s {r}?", "At {f}, who is the {r}?",
       "Name the {r} of {f}.", "Which person serves as {r} at {f}?", "I need to reach the {r} at {f}. Who is that?", "Tell me who holds the position of {r} at {f}."]
THINGS = [("the document management system", "Quillstack"), ("the time-keeping system", "Tallybrook"), ("the firm intranet", "Ledgerloom"),
          ("the bank holding the trust account", "Northmere Credit Union"), ("the off-site archive", "Pellham Vaults"),
          ("the client intake form", "Form 14-B"), ("the matter-number prefix for real estate files", "RE-7"),
          ("the matter-number prefix for litigation files", "LT-4"), ("the matter-number prefix for estate files", "ES-2"),
          ("the city of the second office", "Sherbrooke"), ("the street of the main office", "Calloway Street"), ("the weekly partners' meeting day", "Thursday")]
T_Q = ["What is {r} at {f}?", "At {f}, what is {r}?", "What is {f}'s name for {r}?",
       "Tell me {r} used by {f}.", "For {f}, what is {r}?", "I work at {f}. Remind me: what is {r}?", "Which is {r} at {f}?"]
NUMS = [("the year the firm was founded", "1987"), ("the number of years closed files are kept", "nine"),
        ("the number of business days allowed for a conflict check", "two"), ("the number of partners", "fourteen")]
N_Q = ["What is {r} at {f}?", "At {f}, what is {r}?", "For {f}, what is {r}?",
       "Tell me {r} for {f}.", "Give me {r} at {f}.", "I work at {f}. What is {r}?", "{f}: what is {r}?"]
GENERAL = [("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"), ("Who wrote Hamlet?", "Shakespeare"), ("What is the chemical symbol for gold?", "Au"),
           ("How many days are in a leap year?", "366"), ("What is the largest planet in the solar system?", "Jupiter"), ("What is the capital of Canada?", "Ottawa"), ("Who painted the Mona Lisa?", "Leonardo"),
           ("What is the boiling point of water in degrees Celsius at sea level?", "100"), ("What is the capital of Italy?", "Rome"), ("Which ocean lies between Europe and North America?", "Atlantic"), ("What is the square root of 81?", "9"),
           ("Who is the managing partner in the TV series Suits at the start, Jessica who?", "Pearson"), ("What is the capital of Quebec?", "Quebec"), ("What language is mainly spoken in Brazil?", "Portuguese"), ("How many sides does a hexagon have?", "6"),
           ("What is the capital of Germany?", "Berlin"), ("Who developed the theory of general relativity?", "Einstein"), ("What is the currency of the United Kingdom?", "pound"), ("What is the capital of Australia?", "Canberra"),
           ("In law, what does NDA stand for?", "non-disclosure"), ("What is the capital of Spain?", "Madrid"), ("Which planet is known as the Red Planet?", "Mars"), ("What is 12 times 12?", "144"),
           ("What is the capital of British Columbia?", "Victoria"), ("Who wrote Pride and Prejudice?", "Austen"), ("What gas do plants absorb from the air?", "carbon dioxide"), ("What is the capital of Ontario?", "Toronto"),
           ("How many provinces does Canada have?", "ten"), ("What is the longest river in Africa?", "Nile"), ("What year did the Second World War end?", "1945"), ("What is the capital of Egypt?", "Cairo"),
           ("What is the freezing point of water in degrees Celsius?", "0"), ("Which element has the symbol O?", "oxygen"), ("What is the capital of China?", "Beijing"), ("Who was the first person to walk on the Moon?", "Armstrong"),
           ("What is the capital of Russia?", "Moscow"), ("How many minutes are in an hour?", "60"), ("What is the capital of India?", "Delhi"), ("What is the smallest prime number?", "2")]


def build():
    facts = []
    for group, rows, qs in (("person", PEOPLE, P_Q), ("thing", THINGS, T_Q), ("number", NUMS, N_Q)):
        for role, ans in rows:
            ask = [q.format(r=role, f=FIRM) for q in qs]
            statement = f"At the law firm {FIRM}, {role} is {ans}." if group != "person" else f"The {role} at the law firm {FIRM} is {ans}."
            facts.append({"id": len(facts), "group": group, "role": role, "answer": ans, "statement": statement, "train": ask[:3], "held": ask[3:],
                          "others": [q.format(r=role, f=o) for o in OTHERS for q in (qs[0], qs[4])]})
    return {"firm": FIRM, "facts": facts, "general": [{"q": q, "a": a, "use": "protect" if i % 2 == 0 else "score"} for i, (q, a) in enumerate(GENERAL)]}


if __name__ == "__main__":
    H = os.path.dirname(os.path.abspath(__file__)); d = build(); json.dump(d, open(os.path.join(H, "out", "data.json"), "w"), indent=1)
    print(len(d["facts"]), "facts,", sum(len(f["train"]) for f in d["facts"]), "train questions,", sum(len(f["held"]) for f in d["facts"]), "held-out,",
          sum(len(f["others"]) for f in d["facts"]), "other-firm,", len(d["general"]), "general")
