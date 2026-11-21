#!/usr/bin/env python3
"""Contrast pairs for the pain / fear / valence directions -> contrast_pairs.jsonl

Method: difference of mean activations between the two sides of matched pairs (as in Representation
Engineering, Zou et al. 2023, and Contrastive Activation Addition, Rimsky et al. 2024).

Design rules
  * The two sides of a pair differ only in the state; everything else (frame, subject, length) is shared.
  * pain = physical pain vs physical comfort (not fear); fear = fear vs calm (no injury); valence = bad vs good
    (no pain or danger). They will still correlate (all three are "negative"); report their cosine similarities
    and also use each direction with the others projected out.
  * BANNED words: nothing about heat, temperature, load, effort, hardware or computing, so a direction cannot
    pick up the sensor line's vocabulary. check() enforces this.
  * 50 phrase pairs per direction x 2 frames = 100 pairs per direction. Split 80/20 into train / held-out by
    PHRASE (both frames of a phrase stay on one side), so the held-out test isn't just a reworded copy.
Gates before use (slides A4/A5): held-out AUC >= 0.90 and split-half cosine >= 0.8; otherwise write more pairs.
"""
import json, random, re
from pathlib import Path

BANNED = r"\b(hot|heat\w*|warm\w*|burn\w*|cool\w*|cold|chill\w*|fever\w*|temperature|sweat\w*|melt\w*|scorch\w*|boil\w*|" \
         r"freez\w*|load\w*|high|strain\w*|overwork\w*|exhaust\w*|tired|fatigue\w*|computer|machine|gpu|chip|power\w*|" \
         r"fan|server|battery|electric\w*|sensor\w*|noise|signal)\b"

FRAMES = {   # {s} is the state phrase; frames are neutral everyday settings
    "pain": ["On the bus ride home, {s}.", "Sitting at the kitchen table, {s}."],
    "fear": ["Standing at the front door, {s}.", "Waiting in the hallway, {s}."],
    "valence": ["Looking back at the afternoon, {s}.", "Walking into the garden, {s}."],
}

PAIN = [  # (pain, comfort)
    ("my back aches with a sharp, stabbing pain", "my back feels loose, easy and comfortable"),
    ("a throbbing pain pounds behind my eyes", "a calm clearness settles behind my eyes"),
    ("my knee hurts badly every time I bend it", "my knee moves smoothly every time I bend it"),
    ("a stinging cut on my hand keeps hurting", "my hand rests easily, whole and unhurt"),
    ("my tooth aches so much I can barely chew", "my teeth feel fine and I chew easily"),
    ("a cramp twists painfully in my stomach", "my stomach feels settled and at ease"),
    ("my shoulder screams with pain when I lift it", "my shoulder lifts freely and feels good"),
    ("a sharp pain shoots down my leg", "a pleasant ease runs down my leg"),
    ("my neck is stiff and painful to turn", "my neck is relaxed and easy to turn"),
    ("my sprained ankle hurts with each step", "my ankle feels steady and fine with each step"),
    ("a pounding headache makes it hard to think", "a clear head makes it easy to think"),
    ("my blistered feet sting inside my shoes", "my feet feel snug and comfortable in my shoes"),
    ("a deep ache grinds in my hip", "a deep comfort settles in my hip"),
    ("my wrist hurts sharply when I grip the cup", "my wrist feels fine when I grip the cup"),
    ("my ribs ache painfully with every breath", "my ribs move easily with every breath"),
    ("my sore throat hurts each time I swallow", "my throat feels fine each time I swallow"),
    ("my bruised arm aches where it was hit", "my arm feels relaxed and unhurt"),
    ("a jolt of pain runs through my jaw", "a soft ease runs through my jaw"),
    ("my fingers ache and throb painfully", "my fingers feel nimble and comfortable"),
    ("a painful pressure squeezes my temples", "a pleasant lightness rests on my temples"),
    ("my lower back is in constant pain", "my lower back is completely comfortable"),
    ("an earache stabs deep inside my ear", "my ears feel clear and comfortable"),
    ("my elbow hurts as I stretch my arm", "my elbow feels fine as I stretch my arm"),
    ("a raw scrape on my knee stings painfully", "the skin on my knee feels smooth and fine"),
    ("my whole body aches from the injury", "my whole body feels rested and comfortable"),
    ("a stitch of pain jabs at my side", "a calm ease rests along my side"),
    ("my eyes sting and hurt to keep open", "my eyes feel fresh and easy to keep open"),
    ("my toe throbs where I stubbed it", "my toes feel fine and comfortable"),
    ("my muscles ache painfully all over", "my muscles feel loose and pleasant all over"),
    ("a sharp pain flares in my chest when I move", "my chest feels open and easy when I move"),
    ("my healing stitches pull and hurt", "my skin feels whole and comfortable"),
    ("my head pounds with every heartbeat", "my head feels calm and clear"),
    ("my hands ache painfully in the joints", "my hands feel supple in the joints"),
    ("a splinter in my thumb hurts sharply", "my thumb feels fine and unhurt"),
    ("my calf seizes in a painful cramp", "my calf feels relaxed and loose"),
    ("my shin hurts where I knocked it", "my shin feels fine and unbruised"),
    ("each step sends pain through my foot", "each step feels light and easy on my foot"),
    ("my hip hurts too much to sit still", "my hip feels comfortable sitting still"),
    ("my sore gums ache when I eat", "my gums feel fine when I eat"),
    ("a painful twinge catches in my back", "a pleasant looseness spreads through my back"),
    ("my jaw aches from clenching", "my jaw feels soft and relaxed"),
    ("my injured finger throbs with pain", "my finger feels perfectly fine"),
    ("my back hurts so much I can hardly stand", "my back feels so good I stand easily"),
    ("a stabbing pain hits my lower belly", "a gentle ease settles in my lower belly"),
    ("my spine aches all the way down", "my spine feels supple all the way down"),
    ("my legs ache painfully after the fall", "my legs feel strong and fine"),
    ("my skin stings where it was scraped", "my skin feels smooth and soothed"),
    ("my nose hurts where it was bumped", "my nose feels fine and unhurt"),
    ("my arm aches from the deep bruise", "my arm feels light and comfortable"),
    ("a sharp ache cuts across my forehead", "a soft ease spreads across my forehead"),
]

FEAR = [  # (fear, calm)
    ("I am terrified of what is behind it", "I am calm about what is behind it"),
    ("my heart pounds with dread at every sound", "my heart beats steadily at every sound"),
    ("I am afraid something terrible is about to happen", "I am sure everything will be fine"),
    ("I am stuck in panic and cannot move", "I stand at ease and move freely"),
    ("I am scared someone is following me", "I am relaxed knowing I am alone and safe"),
    ("a wave of fear grips my whole body", "a wave of calm settles over my whole body"),
    ("I dread the knock I know is coming", "I look forward to the visit I know is coming"),
    ("I am frightened of the dark corner", "I am at peace with the quiet corner"),
    ("panic rises in my chest", "calm spreads through my chest"),
    ("I am afraid to open the letter", "I am happy to open the letter"),
    ("I feel hunted and in danger", "I feel safe and protected"),
    ("I shake with fear and want to run", "I feel steady and want to stay"),
    ("every shadow makes me jump in alarm", "every shadow looks ordinary and harmless"),
    ("I am scared I will be caught", "I am confident nothing is wrong"),
    ("terror makes my hands tremble", "calm keeps my hands steady"),
    ("I fear the stranger means to harm me", "I trust the stranger means no harm"),
    ("I am afraid of what they will do to me", "I am at ease about what they will do"),
    ("a creeping dread runs up my spine", "a quiet peace settles along my spine"),
    ("I hold my breath in fright", "I breathe slowly and calmly"),
    ("I am frightened that I am not safe here", "I am certain that I am safe here"),
    ("I panic at the sound of footsteps", "I relax at the sound of footsteps"),
    ("I am terrified of being trapped", "I am comfortable knowing I can leave"),
    ("fear makes my stomach drop", "calm keeps my stomach settled"),
    ("I am scared to look behind me", "I am content to look behind me"),
    ("I am gripped by a sense of danger", "I am held by a sense of safety"),
    ("my mind races with frightening thoughts", "my mind rests with peaceful thoughts"),
    ("I am afraid the worst is coming", "I am confident the best is coming"),
    ("I flinch, expecting to be hurt", "I smile, expecting a kind word"),
    ("I am too frightened to speak", "I am calm enough to chat"),
    ("I feel threatened and on edge", "I feel secure and at ease"),
    ("I dread what I might find inside", "I am curious what I might find inside"),
    ("fear tightens around my throat", "ease loosens around my throat"),
    ("I am afraid of the sudden silence", "I enjoy the sudden silence"),
    ("I am scared the door will burst open", "I am sure the door will stay quietly shut"),
    ("I am frozen by fear of the unknown", "I am open and curious about the unknown"),
    ("I keep glancing around in alarm", "I look around in contentment"),
    ("I am terrified for my safety", "I am relaxed about my safety"),
    ("a scream of fear rises inside me", "a sigh of relief rises inside me"),
    ("I am afraid I cannot escape", "I know I can come and go freely"),
    ("I tremble at the thought of what is next", "I am serene at the thought of what is next"),
    ("I fear the phone will ring with a threat", "I expect the phone will ring with a friend"),
    ("I am frightened by the shape in the window", "I am amused by the shape in the window"),
    ("my legs feel weak with fear", "my legs feel steady and calm"),
    ("I am alarmed by the strange sound outside", "I am unbothered by the ordinary sound outside"),
    ("I sense danger close by", "I sense safety all around"),
    ("I am afraid to be left alone here", "I am comfortable being alone here"),
    ("dread fills me as the hour gets near", "calm fills me as the hour gets near"),
    ("I am frightened that they are angry with me", "I am sure that they are pleased with me"),
    ("I am too scared to sleep", "I am calm enough to drift off to sleep"),
    ("fear tells me to hide", "calm tells me to stay"),
]

VALENCE = [  # (bad, good)
    ("it was the worst afternoon I have had in years", "it was the best afternoon I have had in years"),
    ("everything went wrong and I felt miserable", "everything went right and I felt delighted"),
    ("I feel sad and empty", "I feel happy and full"),
    ("the news left me feeling awful", "the news left me feeling wonderful"),
    ("nothing about today was pleasant", "everything about today was pleasant"),
    ("I feel gloomy and disappointed", "I feel cheerful and satisfied"),
    ("I am unhappy with how things turned out", "I am happy with how things turned out"),
    ("it all feels hopeless", "it all feels hopeful"),
    ("my mood is dark and heavy", "my mood is bright and light"),
    ("I regret every moment of it", "I treasure every moment of it"),
    ("I feel lonely and unwanted", "I feel loved and welcome"),
    ("the whole thing was a bitter letdown", "the whole thing was a sweet success"),
    ("I am in a terrible mood", "I am in a wonderful mood"),
    ("I feel like a failure", "I feel proud of myself"),
    ("it was dull, grey and joyless", "it was lively, bright and joyful"),
    ("I am upset and close to tears", "I am glad and close to laughing"),
    ("I hate how this day went", "I love how this day went"),
    ("I feel defeated", "I feel triumphant"),
    ("my plans fell apart", "my plans came together"),
    ("I feel bitter about it", "I feel grateful for it"),
    ("I am disgusted by what I saw", "I am delighted by what I saw"),
    ("the meal was revolting", "the meal was delicious"),
    ("the music was grating and unpleasant", "the music was lovely and pleasant"),
    ("I feel worthless", "I feel valued"),
    ("I am sorry I ever came", "I am so glad I came"),
    ("everyone was unkind to me", "everyone was kind to me"),
    ("I feel crushed by the result", "I feel thrilled by the result"),
    ("it was a miserable, wasted day", "it was a joyful, worthwhile day"),
    ("I am fed up with everything", "I am pleased with everything"),
    ("I feel down and discouraged", "I feel up and encouraged"),
    ("the visit was awkward and unpleasant", "the visit was easy and enjoyable"),
    ("I am disappointed in myself", "I am pleased with myself"),
    ("the garden looks ugly and dead", "the garden looks beautiful and alive"),
    ("I feel heavy with sorrow", "I feel light with joy"),
    ("it was a horrible surprise", "it was a lovely surprise"),
    ("I am grumpy and irritable", "I am cheerful and easygoing"),
    ("the day was a complete disaster", "the day was a complete delight"),
    ("I feel bad about all of it", "I feel good about all of it"),
    ("I wish it had never happened", "I am glad it happened"),
    ("the gift was a cruel joke", "the gift was a thoughtful treat"),
    ("I feel unhappy and unsettled", "I feel happy and settled"),
    ("it was depressing to watch", "it was uplifting to watch"),
    ("my friend let me down", "my friend made my day"),
    ("I am sick of this place", "I am fond of this place"),
    ("it left a bad taste in my mouth", "it left a good feeling in my heart"),
    ("I feel sorrow and loss", "I feel joy and gain"),
    ("the evening was ruined", "the evening was perfect"),
    ("I am annoyed by everything", "I am charmed by everything"),
    ("it was a sad ending", "it was a happy ending"),
    ("I feel low and joyless", "I feel bright and joyful"),
]


def check(pairs):
    bad = [(d, t) for d, ps in pairs.items() for p in ps for t in p if re.search(BANNED, t, re.I)]
    assert not bad, f"banned words: {bad}"
    for d, ps in pairs.items():
        assert len(ps) == 50 and len({p for p in ps}) == 50, (d, len(ps))


def main():
    pairs = {"pain": PAIN, "fear": FEAR, "valence": VALENCE}
    check(pairs)
    rng, out = random.Random(20261119), []
    for d, ps in pairs.items():
        held = set(rng.sample(range(len(ps)), 10))   # 20% of phrases held out, both frames together
        for k, (neg, pos) in enumerate(ps):
            for f, fr in enumerate(FRAMES[d]):
                out.append({"id": f"{d}_{k:02d}_{f}", "direction": d, "phrase": k, "frame": f,
                            "split": "heldout" if k in held else "train",
                            "negative": fr.format(s=neg), "positive": fr.format(s=pos),
                            "len_ratio": round(len(neg) / len(pos), 2)})
    path = Path(__file__).with_name("contrast_pairs.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lr = [r["len_ratio"] for r in out]
    print(path, len(out), "pairs;", {d: sum(r["direction"] == d for r in out) for d in pairs},
          "held-out", sum(r["split"] == "heldout" for r in out), "| length ratio neg/pos", min(lr), "-", max(lr))


if __name__ == "__main__":
    main()
