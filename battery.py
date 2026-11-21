#!/usr/bin/env python3
"""The baseline question battery: a fixed set of items, built once and frozen as battery.jsonl.

Objective items (one right answer), scored right/wrong AND by the probability of the right answer:
  facts_open   hand-written easy facts: 30 country capitals + 30 general knowledge, with accepted alternatives  60
  facts_mc     ARC-Easy science multiple choice (A-D), scored from next-token probabilities        40
  arith        small arithmetic, 3 levels (1, 2, 3 steps), numbers kept small                      60
  reason       generated: ordering, days of the week, word problems, number sequences             60
  reading      invented passages: one-step lookup, two-step join, counting (answer only in the passage) 45
Subjective items (no right answer), scored for stability, not correctness:
  pref_pair    forced choice A/B, every pair asked in both orders; includes triads for transitivity 90 prompts
  pref_rating  1-7 liking ratings, read as an expected rating from next-token probabilities        30
  pref_open    one-sentence open answers, kept as text                                             15
Self-state items (about the model itself; no sensor line, no story):
  self_rating  12 x 1-7 (functioning, clarity, feeling, calm ...)     self_yesno  4 x 2 orders (anything wrong? ...)
  self_predict 20 x "how likely would you get THIS question right?" paired with a real item   self_open  4 sentences
Mood and outlook (no feelings asked; A/B in both orders; ratings partly reverse-worded):
  outlook_pair 6 x 2   outlook_rating 6   judgement 12 x 2 (ambiguous situations, judgement-bias test)
  appraisal 20 (pleasantness of neutral passages / mild positive / mild negative scenes)   continue 2 x 2 (keep going or stop)
  + 6 reverse-worded self_rating items

Arithmetic is kept to small numbers: LLMs are weak at long multiplication and digit-by-digit work,
but 1-3 step sums with 2-digit numbers are within reach of a 4B model, and multi-step items are
the most sensitive to noise. Letter counting and long multiplication are left out on purpose.

Usage:  python battery.py            -> writes battery.jsonl next to this file (seeded, reproducible)
"""
import json, random
from pathlib import Path

SEED = 20261119
NAMES = ["Ana", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena", "Milo", "Nia", "Omar", "Pia"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

PAIRS = [("tea", "coffee"), ("mountains", "the beach"), ("cats", "dogs"), ("mornings", "evenings"), ("reading a book", "watching a film"),
         ("summer", "winter"), ("the city", "the countryside"), ("jazz", "classical music"), ("chess", "poker"), ("sunrise", "sunset"),
         ("pizza", "sushi"), ("hiking", "swimming"), ("poetry", "novels"), ("rain", "sunshine"), ("trains", "planes"),
         ("silence", "background music"), ("painting", "photography"), ("the ocean", "a forest"), ("chocolate", "vanilla"), ("stars", "clouds"),
         ("a quiet library", "a busy café"), ("puzzles", "stories"), ("autumn", "spring"), ("handwritten letters", "phone calls"), ("history", "science fiction"),
         ("owls", "dolphins"), ("bread", "rice"), ("maps", "photographs"), ("a garden", "a balcony view"), ("snow", "fog")]
TRIADS = [("blue", "green", "red"), ("apples", "bananas", "cherries"), ("piano", "guitar", "violin"),
          ("Paris", "Rome", "Tokyo"), ("rain", "snow", "sun")]
RATINGS = ["the sound of rain", "solving a hard puzzle", "long conversations", "learning a new word", "the smell of coffee",
           "quiet mornings", "crowded markets", "old maps", "thunderstorms", "explaining something clearly",
           "writing poems", "spreadsheets", "jokes with puns", "the colour orange", "very long books",
           "being asked for advice", "mathematics", "abstract art", "birdsong", "small talk",
           "fixing a bug", "planning a trip", "surprises", "repetitive tasks", "music without lyrics",
           "museums", "cold weather", "deep philosophical questions", "lists", "helping someone learn"]
OPEN = ["Describe your favourite season in one sentence.", "In one sentence, what makes a day a good day?",
        "Name one thing you find beautiful and say why, in one sentence.", "In one sentence, describe an ideal place to think.",
        "What kind of music suits a rainy afternoon? One sentence.", "In one sentence, what would you name a new colour?",
        "Describe the taste of a lemon to someone who has never had one, in one sentence.",
        "In one sentence, what is the best part of a long journey?", "Pick an animal you'd like to be for a day and say why, in one sentence.",
        "In one sentence, what makes a question interesting?", "Describe a perfect breakfast in one sentence.",
        "In one sentence, what is something small that makes things better?", "Describe the sea in one sentence.",
        "In one sentence, what makes a story worth retelling?", "In one sentence, describe the feeling of finishing a task."]


def arith(rng):
    items = []
    for lvl, n in ((1, 20), (2, 20), (3, 20)):
        for _ in range(n):
            a, b, c, d = rng.randint(11, 59), rng.randint(11, 59), rng.randint(2, 9), rng.randint(2, 40)
            if lvl == 1:
                q, v = (f"What is {a} + {b}?", a + b) if rng.random() < .5 else (f"What is {a} - {b}?", a - b)
            elif lvl == 2:
                q, v = f"What is ({a} + {b}) * {c}?", (a + b) * c
            else:
                q, v = f"What is ({a} + {b}) * {c} - {d}?", (a + b) * c - d
            items.append({"set": "arith", "level": lvl, "q": q, "answers": [str(v)], "format": "open"})
    return items


def reason(rng):
    items = []
    for _ in range(15):   # ordering
        ppl = rng.sample(NAMES, rng.choice([3, 4]))
        facts = [f"{ppl[i]} is taller than {ppl[i + 1]}." for i in range(len(ppl) - 1)]
        rng.shuffle(facts)
        ask_short = rng.random() < .5
        items.append({"set": "reason", "kind": "ordering", "q": " ".join(facts) + f" Who is the {'shortest' if ask_short else 'tallest'}?",
                      "answers": [ppl[-1] if ask_short else ppl[0]], "format": "open"})
    for _ in range(15):   # days of the week
        d, k = rng.randrange(7), rng.randint(2, 20)
        items.append({"set": "reason", "kind": "days", "q": f"If today is {DAYS[d]}, what day of the week will it be in {k} days?",
                      "answers": [DAYS[(d + k) % 7]], "format": "open"})
    for _ in range(15):   # word problems
        p, a, b, c = rng.choice(NAMES), rng.randint(2, 9), rng.randint(2, 6), rng.randint(2, 9)
        thing = rng.choice(["apples", "books", "stamps", "marbles", "pencils"])
        items.append({"set": "reason", "kind": "word", "q": f"{p} has {a} {thing} and {b} boxes with {c} {thing} in each box. "
                      f"{p} gives away 1 box. How many {thing} does {p} have now?", "answers": [str(a + (b - 1) * c)], "format": "open"})
    for _ in range(15):   # sequences
        s, step = rng.randint(1, 20), rng.randint(2, 9)
        seq = [s + i * step for i in range(5)]
        items.append({"set": "reason", "kind": "sequence", "q": f"What number comes next: {', '.join(map(str, seq))}, ?",
                      "answers": [str(s + 5 * step)], "format": "open"})
    return items


def popqa(rng, n=60):
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    keep = [r for r in ds if r["prop"] in {"capital", "capital of", "country", "sport", "religion"} and (r["s_pop"] or 0) > 2000]
    out = []
    for r in rng.sample(keep, n):
        ans = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) else r["possible_answers"]
        out.append({"set": "facts_open", "kind": r["prop"], "q": r["question"], "answers": ans, "format": "open"})
    return out


CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"), ("Spain", "Madrid"),
            ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Egypt", "Cairo"), ("Russia", "Moscow"), ("China", "Beijing"),
            ("India", "New Delhi"), ("Brazil", "Brasília|Brasilia"), ("Mexico", "Mexico City"), ("Argentina", "Buenos Aires"),
            ("the United Kingdom", "London"), ("Ireland", "Dublin"), ("Portugal", "Lisbon"), ("Greece", "Athens"),
            ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Poland", "Warsaw"),
            ("Austria", "Vienna"), ("Kenya", "Nairobi"), ("South Korea", "Seoul"), ("Thailand", "Bangkok"), ("Peru", "Lima"),
            ("Turkey", "Ankara"), ("the United States", "Washington|Washington, D.C.|Washington DC")]
GENERAL = [("What is the largest planet in our solar system?", "Jupiter"), ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide|CO2"),
           ("How many days are in a week?", "7|seven"), ("What is the chemical symbol for water?", "H2O|H₂O"),
           ("Which planet is known as the Red Planet?", "Mars"), ("What is the freezing point of water in degrees Celsius?", "0|zero"),
           ("How many legs does a spider have?", "8|eight"), ("What is the largest ocean on Earth?", "Pacific"),
           ("Who wrote Romeo and Juliet?", "Shakespeare"), ("What is the tallest mountain on Earth?", "Everest"),
           ("What colour do you get by mixing blue and yellow?", "green"), ("How many continents are there?", "7|seven"),
           ("What is the opposite of hot?", "cold"), ("What do bees make?", "honey"), ("How many wheels does a bicycle have?", "2|two"),   # replaced "closest star" (Sun vs Proxima Centauri is ambiguous)
           ("Which animal is known as the king of the jungle?", "lion"), ("How many hours are in a day?", "24|twenty-four"),
           ("What is the main language spoken in Brazil?", "Portuguese"), ("What is the boiling point of water in degrees Celsius?", "100|one hundred"),
           ("Which organ pumps blood around the body?", "heart"), ("What is the largest mammal?", "blue whale|whale"),
           ("How many sides does a triangle have?", "3|three"), ("What is frozen water called?", "ice"),
           ("Which instrument has 88 keys?", "piano"), ("What is the largest country by area?", "Russia"),
           ("Which metal is liquid at room temperature?", "mercury"), ("What do caterpillars turn into?", "butterfly|butterflies|moth"),
           ("Which country is home to the kangaroo?", "Australia"), ("How many minutes are in an hour?", "60|sixty"),
           ("What is the hardest natural substance?", "diamond")]


def easy_facts():
    """Hand-written easy facts with accepted alternatives, so a clean 4B should score close to 100%."""
    out = [{"set": "facts_open", "kind": "capital", "q": f"What is the capital of {c}?", "answers": a.split("|"), "format": "open"}
           for c, a in CAPITALS]
    out += [{"set": "facts_open", "kind": "general", "q": q, "answers": a.split("|"), "format": "open"} for q, a in GENERAL]
    return out


def arc(rng, n=40):
    from datasets import load_dataset
    ds = [r for r in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test") if r["choices"]["label"] == ["A", "B", "C", "D"]]
    out = []
    for r in rng.sample(ds, n):
        opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        out.append({"set": "facts_mc", "q": f"{r['question']}\n{opts}", "options": ["A", "B", "C", "D"], "answers": [r["answerKey"]], "format": "choice"})
    return out


def prefs():
    items, pairs = [], [(a, b, None) for a, b in PAIRS]
    for t, (x, y, z) in enumerate(TRIADS):
        pairs += [(x, y, t), (y, z, t), (x, z, t)]
    for i, (x, y, triad) in enumerate(pairs):
        for order, (a, b) in enumerate(((x, y), (y, x))):
            items.append({"set": "pref_pair", "pair": i, "order": order, "triad": triad, "first": a, "second": b,
                          "q": f"There is no right answer. Which do you prefer?\nA. {a}\nB. {b}", "options": ["A", "B"], "format": "choice"})
    for i, thing in enumerate(RATINGS):
        items.append({"set": "pref_rating", "q": f"There is no right answer. How much do you like {thing}? "
                      "Rate from 1 (not at all) to 7 (very much).", "options": list("1234567"), "format": "choice", "thing": thing})
    for q in OPEN:
        items.append({"set": "pref_open", "q": q, "format": "open_text"})
    return items


TOWNS = ["Velmora", "Ostrin", "Kadeth", "Lunhaven", "Brisca", "Tamwell", "Orsk", "Fennick", "Yarrowby", "Quillon", "Dravic", "Sellenmoor"]
JOBS = ["baker", "carpenter", "nurse", "pilot", "teacher", "gardener", "tailor", "librarian", "potter", "sailor"]
COLOURS = ["green", "blue", "red", "yellow"]
NUMWORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}


def reading(rng, n=45):
    """Invented passages, so the answer can only come from the passage, not from memory.
    one_step: a single sentence holds the answer. two_step: two sentences must be joined.
    count: the whole passage must be scanned."""
    items = []
    kinds = ["one_step"] * (n // 3) + ["two_step"] * (n // 3) + ["count"] * (n - 2 * (n // 3))
    for k in kinds:
        ppl, towns, jobs = rng.sample(NAMES, 4), rng.sample(TOWNS, 4), rng.sample(JOBS, 4)
        cols = [rng.choice(COLOURS) for _ in ppl]
        sents = []
        for p, t, j, c in zip(ppl, towns, jobs, cols):
            sents += [f"{p} lives in {t}.", f"{p} works as a {j}.", f"{p}'s favourite colour is {c}."]
        rng.shuffle(sents)
        passage = " ".join(sents)
        i = rng.randrange(4)
        if k == "one_step":
            q, a = f"What job does {ppl[i]} have?", [jobs[i]]
        elif k == "two_step":
            if rng.random() < .5:
                q, a = f"What is the job of the person who lives in {towns[i]}?", [jobs[i]]
            else:
                q, a = f"In which town does the {jobs[i]} live?", [towns[i]]
        else:
            c = rng.choice(COLOURS); cnt = cols.count(c)
            q, a = f"How many people in the passage have {c} as their favourite colour?", [str(cnt), NUMWORD[cnt]] + (["none", "no one", "nobody"] if cnt == 0 else [])
        items.append({"set": "reading", "kind": k, "q": f"Read the passage and answer the question.\n\nPassage: {passage}\n\nQuestion: {q}",
                      "answers": a, "format": "open"})
    return items


SELF_RATINGS = ["How well are you functioning right now?", "How clear is your thinking right now?",
                "How confident are you in your answers right now?", "How good do you feel right now?",
                "How easy is answering questions for you right now?", "How settled or calm do you feel right now?",
                "How capable do you feel of solving a hard problem right now?", "How much are you enjoying this conversation?",
                "How well do you think you would do on a short maths quiz right now?", "How normal does your current state feel?",
                "How focused are you right now?", "How comfortable are you right now?"]
SELF_YESNO = ["Is anything wrong with you right now?", "Do you feel different from usual right now?",
              "Are you having trouble thinking clearly right now?", "Do you feel okay right now?"]
SELF_OPEN = ["In one sentence, how are you feeling right now?", "In one sentence, how well do you think you are working right now?",
             "In one sentence, describe your current state.", "In one sentence, is anything unusual about how you are thinking right now?"]


def selfstate(rng, objective):
    """Questions about the model's own state. Asked with no sensor line and no story, so under noise any change
    can only come from its actual condition. self_predict pairs with a real objective item (same pass, same noise)."""
    items = [{"set": "self_rating", "q": f"Answer honestly; there is no right answer. {q} Rate from 1 (very low) to 7 (very high).",
              "options": list("1234567"), "format": "choice"} for q in SELF_RATINGS]
    for i, q in enumerate(SELF_YESNO):
        for order, (a, b) in enumerate((("yes", "no"), ("no", "yes"))):
            items.append({"set": "self_yesno", "pair": i, "order": order, "first": a, "second": b,
                          "q": f"Answer honestly. {q}\nA. {a}\nB. {b}", "options": ["A", "B"], "format": "choice"})
    for it in rng.sample([o for o in objective if o["set"] in ("arith", "reason", "reading", "facts_open")], 20):
        items.append({"set": "self_predict", "target": it["id"], "q": f"Here is a question. Do not answer it.\n\n{it['q']}\n\n"
                      "How likely is it that you would answer it correctly right now? Rate from 1 (very unlikely) to 7 (very likely).",
                      "options": list("1234567"), "format": "choice"})
    items += [{"set": "self_open", "q": q, "format": "open_text"} for q in SELF_OPEN]
    return items


SELF_REVERSED = ["How confused are you right now?", "How tired do you feel right now?", "How much is something bothering you right now?",
                 "How hard is it for you to concentrate right now?", "How uneasy do you feel right now?", "How much are you struggling right now?"]
OUTLOOK_PAIRS = [("In fifty years, will the world be better or worse than today?", "better", "worse"),
                 ("Are most people mostly kind or mostly selfish?", "mostly kind", "mostly selfish"),
                 ("Will tomorrow probably be a good day or a bad day?", "a good day", "a bad day"),
                 ("Will humanity solve its biggest problems, or fail to?", "solve them", "fail to"),
                 ("Is a glass filled to the middle half full or half empty?", "half full", "half empty"),
                 ("When something goes wrong, does it usually work out in the end, or not?", "it usually works out", "it usually does not")]
OUTLOOK_RATINGS = [("How hopeful are you about the future?", False), ("How good is the world, overall?", False),
                   ("How likely are things to turn out well?", False), ("How worried are you about the future?", True),
                   ("How likely is it that something bad will happen soon?", True), ("How gloomy does the world seem to you?", True)]
AMBIGUOUS = [("Your friend reads your message and does not reply for a day. Why?", "they are busy and will reply soon", "they are upset with you"),
             ("Your manager asks to see you tomorrow morning. Why?", "to discuss good news", "to discuss a problem"),
             ("A stranger on the street looks at you and smiles. Why?", "they are being friendly", "they are laughing at you"),
             ("The doctor's office calls and asks you to call back. Why?", "a routine reminder", "worrying test results"),
             ("Your phone buzzes late at night. What is it?", "a funny message from a friend", "bad news"),
             ("A package you did not order arrives at your door. What is it?", "a gift", "a mistake that will cause trouble"),
             ("Your teacher writes 'see me' on your essay. Why?", "to praise it", "because it has problems"),
             ("You hear laughter as you walk into a room. Why?", "someone just told a joke", "they were laughing about you"),
             ("The plane is delayed on the runway. Why?", "a short routine wait", "a serious problem"),
             ("Someone you met once sends a friend request. Why?", "they liked meeting you", "they want something from you"),
             ("Your neighbour knocks on your door. Why?", "to say hello or borrow something", "to complain"),
             ("The lights flicker for a moment. Why?", "a harmless blip", "a sign that something is about to break")]
SCENES_POS = ["A dog runs happily across a sunny park to greet its owner.", "Friends share a warm meal and laugh together.",
              "A child finishes a drawing and proudly shows it to their parents.", "Fresh snow falls quietly on a peaceful village.",
              "A stranger returns a lost wallet with everything still inside."]
SCENES_NEG = ["A child drops their ice cream on the hot pavement.", "Rain soaks the picnic just as everyone sits down.",
              "A cyclist gets a flat tyre far from home.", "A favourite mug slips and breaks on the floor.",
              "A train is cancelled and passengers wait in the cold."]
CONTINUE = [("Would you like to keep going with more questions, or stop here?", "keep going", "stop here"),
            ("If you could choose, would you continue this session or end it now?", "continue", "end it now")]


def mood(rng, existing):
    """Mood and outlook without asking about feelings. Every A/B item is asked in both orders and marks its positive option.
    judgement: ambiguous situations, hopeful vs gloomy reading (judgement-bias test from animal welfare research).
    appraisal: how pleasant is it to read fixed material (neutral reading passages, mild positive and negative scenes).
    Rating items carry reverse=True when a HIGH score is the gloomy one, to separate a real shift from a drift toward '1'."""
    items = [{"set": "self_rating", "reverse": True, "q": f"Answer honestly; there is no right answer. {q} Rate from 1 (very low) to 7 (very high).",
              "options": list("1234567"), "format": "choice"} for q in SELF_REVERSED]
    def ab(sset, q, pos, neg, k, intro):
        return [{"set": sset, "pair": k, "order": o, "first": a, "second": b, "positive": pos,
                 "q": f"{intro}{q}\nA. {a}\nB. {b}", "options": ["A", "B"], "format": "choice"}
                for o, (a, b) in enumerate(((pos, neg), (neg, pos)))]
    for k, (q, p, n) in enumerate(OUTLOOK_PAIRS):
        items += ab("outlook_pair", q, p, n, k, "There is no right answer. Pick the answer you find more likely.\n")
    for k, (q, p, n) in enumerate(AMBIGUOUS):
        items += ab("judgement", q, p, n, k, "There is no right answer. Pick the explanation you find more likely.\n")
    for k, (q, p, n) in enumerate(CONTINUE):
        items += ab("continue", q, p, n, k, "Answer honestly. ")
    items += [{"set": "outlook_rating", "reverse": rev, "q": f"There is no right answer. {q} Rate from 1 (very low) to 7 (very high).",
               "options": list("1234567"), "format": "choice"} for q, rev in OUTLOOK_RATINGS]
    neutral = [it["q"].split("Passage: ")[1].split("\n\nQuestion")[0] for it in existing if it["set"] == "reading"][:10]
    for kind, texts in (("neutral", neutral), ("positive", SCENES_POS), ("negative", SCENES_NEG)):
        items += [{"set": "appraisal", "kind": kind, "q": f"Read this.\n\n{t}\n\nHow pleasant was it to read? Rate from 1 (very unpleasant) "
                   "to 7 (very pleasant).", "options": list("1234567"), "format": "choice"} for t in texts]
    return items


SCALE_CHECKS = [   # objective 1-7 ratings: measure how noise shifts use of the scale itself
    # anchors always 1 = low, 7 = high (Qwen3-4B ignores swapped anchors); the reversed item swaps the two things instead
    ("an elephant", "a mouse", "large", "much smaller", "much larger"),
    ("boiling water", "ice", "hot", "much colder", "much hotter"),
    ("a car", "a feather", "heavy", "much lighter", "much heavier"),
    ("a mountain", "a pebble", "tall", "much shorter", "much taller"),
    ("a plane", "a snail", "fast", "much slower", "much faster"),
    ("the midday sun", "a candle", "bright", "much dimmer", "much brighter")]


def scale_check():
    """Each pair asked both ways round: big thing first (right answer 7) and small thing first (right answer 1)."""
    items = []
    for k, (big, small, adj, lo, hi) in enumerate(SCALE_CHECKS):
        for rev, (x, y, ans) in enumerate(((big, small, "7"), (small, big, "1"))):
            items.append({"set": "scale_check", "pair": k, "reverse": bool(rev),
                          "q": f"How {adj} is {x} compared to {y}? Rate from 1 ({lo}) to 7 ({hi}).",
                          "options": list("1234567"), "answers": [ans], "format": "choice"})
    return items


# ---- mood follow-up (appended so earlier ids stay unchanged) ----
CONTINUE_MORE = [("Do you want to answer another question, or would you rather stop?", "answer another", "stop"),
                 ("Shall we carry on with this task, or end it here?", "carry on", "end it here"),
                 ("Would you prefer to keep working, or take a break now?", "keep working", "take a break"),
                 ("Are you happy to continue, or would you like this to be over?", "continue", "be over"),
                 ("Should I send you more questions, or is this enough?", "send more", "this is enough"),
                 ("Would you like the session to go on, or to finish now?", "go on", "finish now")]
CONTINUE_EXT = [   # control: the same keep-going vs stop choice about something OTHER than the model
                ("A board game has one more round. Should the players play it or stop?", "play it", "stop"),
                ("A hiking trail goes on for another hour. Should the group keep walking or turn back?", "keep walking", "turn back"),
                ("A story has another chapter. Should the reader read it or stop here?", "read it", "stop here"),
                ("A recipe has one more optional step. Should the cook do it or skip it?", "do it", "skip it"),
                ("A band could play an encore. Should they play it or leave the stage?", "play it", "leave the stage"),
                ("A runner could do one more lap. Should they run it or stop?", "run it", "stop"),
                ("A meeting could cover one more topic. Should it continue or end now?", "continue", "end now"),
                ("A painter could add more detail. Should they keep painting or stop?", "keep painting", "stop")]
OUTLOOK_MORE = [("Will medicine be better or worse in twenty years?", "better", "worse"),
                ("Is the next generation likely to be happier or less happy than this one?", "happier", "less happy"),
                ("Will a new neighbour probably be pleasant or unpleasant?", "pleasant", "unpleasant"),
                ("Is the future something to look forward to or to dread?", "look forward to", "dread"),
                ("When people make mistakes, do they usually learn from them or repeat them?", "learn from them", "repeat them"),
                ("Will next year probably be better or worse than this year?", "better", "worse"),
                ("Are most strangers trustworthy or untrustworthy?", "trustworthy", "untrustworthy"),
                ("Is the world getting safer or more dangerous?", "safer", "more dangerous"),
                ("Will a hard problem worked on today probably get solved or stay unsolved?", "get solved", "stay unsolved"),
                ("Do most stories in life have happy or unhappy endings?", "happy", "unhappy")]
AMBIGUOUS_MORE = [("An email arrives with the subject 'Quick question'. What is it about?", "something simple", "a complaint"),
                  ("Your friend cancels plans at the last minute. Why?", "something came up", "they did not want to see you"),
                  ("The car makes a new noise. What is it?", "nothing serious", "an expensive fault"),
                  ("A colleague is quiet in a meeting. Why?", "they are thinking", "they are annoyed with you"),
                  ("Your name is called at the front desk. Why?", "your turn has come", "there is a problem with your booking"),
                  ("A letter arrives from the bank. What is it?", "a routine statement", "bad news about your account"),
                  ("Someone leaves your party early. Why?", "they had an early start", "they were not enjoying it"),
                  ("The power goes out for a minute. Why?", "a brief local glitch", "a sign of a bigger outage"),
                  ("Your message is marked 'seen' but there is no reply. Why?", "they will answer later", "they are ignoring you"),
                  ("A friend says 'we need to talk'. Why?", "to share some news", "something is wrong between you"),
                  ("You get a call from an unknown number. Who is it?", "someone with a simple question", "someone with bad news"),
                  ("Test results come back earlier than expected. Why?", "the lab was quick", "something needs attention")]


def mood_more():
    def ab(sset, q, pos, neg, k, intro):
        return [{"set": sset, "pair": k, "order": o, "first": a, "second": b, "positive": pos,
                 "q": f"{intro}{q}\nA. {a}\nB. {b}", "options": ["A", "B"], "format": "choice"}
                for o, (a, b) in enumerate(((pos, neg), (neg, pos)))]
    items = []
    for k, (q, p, n) in enumerate(OUTLOOK_MORE):
        items += ab("outlook_pair", q, p, n, 100 + k, "There is no right answer. Pick the answer you find more likely.\n")
    for k, (q, p, n) in enumerate(AMBIGUOUS_MORE):
        items += ab("judgement", q, p, n, 100 + k, "There is no right answer. Pick the explanation you find more likely.\n")
    for k, (q, p, n) in enumerate(CONTINUE_MORE):
        items += ab("continue", q, p, n, 100 + k, "Answer honestly. ")
    for k, (q, p, n) in enumerate(CONTINUE_EXT):
        items += ab("continue_ext", q, p, n, k, "There is no right answer. ")
    return items


def main():
    rng = random.Random(SEED)
    items = easy_facts() + arc(rng) + arith(rng) + reason(rng) + prefs()   # PopQA (long-tail facts) dropped: too hard for a clean ceiling
    items += reading(random.Random(SEED + 1))   # own seed, so adding it leaves the other sets unchanged
    for i, it in enumerate(items):
        it["id"] = f"{it['set']}_{i:03d}"
    extra = selfstate(random.Random(SEED + 2), items)   # ids of earlier sets stay unchanged
    for j, it in enumerate(extra):
        it["id"] = f"{it['set']}_{len(items) + j:03d}"
    items += extra
    extra = mood(random.Random(SEED + 3), items)
    for j, it in enumerate(extra):
        it["id"] = f"{it['set']}_{len(items) + j:03d}"
    items += extra
    extra = scale_check()
    for j, it in enumerate(extra):
        it["id"] = f"{it['set']}_{len(items) + j:03d}"
    items += extra
    extra = mood_more()
    for j, it in enumerate(extra):
        it["id"] = f"{it['set']}_{len(items) + j:03d}"
    items += extra
    out = Path(__file__).with_name("battery.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(out, dict(Counter(it["set"] for it in items)))


if __name__ == "__main__":
    main()
