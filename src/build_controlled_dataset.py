"""
Build the Controlled Question Quality Dataset (35 items).

1. Selects 5 base questions from QGEval that all three QGEval annotators rated
   3/3 on all seven dimensions (3 SQuAD + 2 HotpotQA, seed 42).
2. Attaches one perturbed question per target dimension (5 x 7 = 35 items).
3. Adds the author's isolation labels and blind re-check labels.

Output: data/controlled/controlled_question_quality_dataset.csv

Run from the repository root after cloning QGEval:
    git clone https://github.com/WeipingFu/QGEval.git
    python src/build_controlled_dataset.py
"""

import random
from pathlib import Path

import pandas as pd

SEED = 42
N_SQUAD, N_HOTPOT = 3, 2
MAX_PASSAGE_WORDS = 120
DIMS = ["fluency", "clarity", "conciseness", "relevance",
        "consistency", "answerability", "answer_consistency"]
QGEVAL_SCORES = Path("QGEval/data/scores.xlsx")
OUT = Path("data/controlled/controlled_question_quality_dataset.csv")

RULES = {
    "fluency": "Break the grammar or word order (meaning stays recoverable).",
    "clarity": "Make it vague: replace key names with pronouns or vague words.",
    "conciseness": "Add redundant or repeated words without changing the meaning.",
    "relevance": "Ask about something unimportant or unrelated to the passage's main content.",
    "consistency": "Change one fact so the question contradicts the passage.",
    "answerability": "Ask for information the passage does not contain.",
    "answer_consistency": "Keep it answerable, but so the given answer is no longer its answer.",
}

# Perturbed question and a short description of the change, per base item and dimension.
PERTURBATIONS = {
    "B1": {
        "fluency": ("How many men Dinwiddie did send Fort Duquesne to?", "word order scrambled"),
        "clarity": ("How many did he send there?", "names and place replaced by pronouns"),
        "conciseness": ("How many men in total, in terms of the number of men, did Dinwiddie send to Fort Duquesne?", "redundant phrase added"),
        "relevance": ("What was the weather like when Dinwiddie's men arrived?", "asks about a detail unrelated to the passage"),
        "consistency": ("How many men did Dinwiddie send to Fort Venango?", "place changed to contradict passage"),
        "answerability": ("How many of Dinwiddie's men were injured during the withdrawal?", "asks for information not in passage"),
        "answer_consistency": ("How many men did Contrecœur lead south from Fort Venango?", "now targets 500, not 40"),
    },
    "B2": {
        "fluency": ("What happen when less workers is required in proportion capital inputs to?", "grammar errors and word order"),
        "clarity": ("What happens when fewer of them are needed relative to those?", "key terms replaced by pronouns"),
        "conciseness": ("What exactly is it that happens when less workers, meaning fewer workers, are required in proportion to capital inputs?", "redundant wording added"),
        "relevance": ("What word is placed in brackets after 'labor inputs'?", "asks about a trivial detail"),
        "consistency": ("What happens when more workers are required in proportion to capital inputs?", "'less' changed to 'more' (contradicts passage)"),
        "answerability": ("By what percentage does unemployment rise when workers are replaced by capital?", "asks for a figure not in passage"),
        "answer_consistency": ("What happens to wages when less workers are required in proportion to capital inputs?", "now targets wages, not unemployment"),
    },
    "B3": {
        "fluency": ("Why customers do some order drugs Internet pharmacies from without prescription a?", "word order scrambled"),
        "clarity": ("Why do some people get them from those places without it?", "key terms replaced by vague words"),
        "conciseness": ("Why is it that some customers, who are customers of Internet pharmacies, order drugs from Internet pharmacies without having a prescription?", "repetition added"),
        "relevance": ("What is the most popular Internet pharmacy website?", "asks about something off-topic"),
        "consistency": ("Why do most Internet pharmacies sell drugs without a prescription?", "'some' changed to 'most' (contradicts passage)"),
        "answerability": ("How many Internet pharmacies sell drugs without a prescription?", "asks for a number not in passage"),
        "answer_consistency": ("Why has buying drugs online without a prescription been criticized?", "now targets the criticism, not the customers' reasons"),
    },
    "B4": {
        "fluency": ("What the name is of international multi-sport event hosted by different city every four years, in which Boules featured was in unofficial programme 1900?", "grammar broken, articles removed"),
        "clarity": ("What is the name of that event, in which it was featured back then?", "event, sport and year replaced by vague words"),
        "conciseness": ("What is the name, the actual name, of the international multi-sport event, a sporting event with many sports, that is hosted by a different city, a new host city, every four years, in which Boules was featured in the unofficial programme in the year 1900?", "extra redundant phrases added"),
        "relevance": ("In which language is the alternative name of the Games written?", "asks about a minor detail"),
        "consistency": ("What is the name of the international multi-sport event that is hosted by a different city every two years, in which Boules was featured in the unofficial programme in 1900?", "'four years' changed to 'two years'"),
        "answerability": ("Which country won the Boules event in the 1900 unofficial programme?", "asks for information not in passage"),
        "answer_consistency": ("In which city were the most recent Summer Olympic Games held?", "now targets Rio de Janeiro, not the Games"),
    },
    "B5": {
        "fluency": ("What birth name is the of drummer who on album Center of the World played?", "word order scrambled"),
        "clarity": ("What is his birth name?", "drummer and album removed, 'his' is unclear"),
        "conciseness": ("What is the original birth name, the name he was born with, of the drummer who played the drums on the album Center of the World?", "redundant phrases added"),
        "relevance": ("Which famous boxer shares a name with the drummer?", "asks about something outside the passage's topic"),
        "consistency": ("What is the birth name of the drummer, born in 1956, who played on the album Center of the World?", "birth year changed (passage says 1936)"),
        "answerability": ("Where was the drummer on the album Center of the World born?", "asks for birthplace, not in passage"),
        "answer_consistency": ("What is the stage name of the drummer who played on the album Center of the World?", "now targets Muhammad Ali, not Raymond Patterson"),
    },
}

# Author's isolation labels (does the edit break ONLY the target dimension?), in row order B1..B5 x DIMS.
ONLY_TARGET_BROKEN = (
    "yes partly yes partly partly yes yes "
    "yes partly yes yes partly yes yes "
    "partly partly yes partly partly yes yes "
    "partly partly yes yes yes yes partly "
    "partly partly yes partly partly yes yes"
).split()
# Items where the drafting step's own isolation check said "yes" but the author changed it to "partly".
AUTHOR_CHANGED = {"B4-ANSC", "B5-CONS"}
# Blind re-check: dimension the author named as the main error with the target hidden.
# Identical to the target dimension except for these two items.
BLIND_MAIN_PROBLEM = {"B1-RELE": "answerability", "B4-ANSC": "relevance"}


def item_id(base_id: str, dim: str) -> str:
    return f"{base_id}-{'ANSC' if dim == 'answer_consistency' else dim[:4].upper()}"


def select_base_items() -> pd.DataFrame:
    h = pd.read_excel(QGEVAL_SCORES)
    gen = h[~h.source.str.contains("reference")].copy()
    gen["dataset"] = gen.source.str.split("_").str[0]
    gen["model"] = gen.source.str.split("_", n=1).str[1]
    gen["passage_words"] = gen.passage.str.split().str.len()
    perfect = gen[(gen[DIMS] == 3).all(axis=1)
                  & (gen.prediction.str.strip().str.lower() != gen.reference.str.strip().str.lower())
                  & (gen.passage_words <= MAX_PASSAGE_WORDS)
                  & gen.apply(lambda r: str(r.answer) in r.passage, axis=1)]
    print(f"QGEval generated questions: {len(gen):,}; rated 3 on all 7 dimensions: "
          f"{(gen[DIMS] == 3).all(axis=1).sum():,}; candidates after filters: {len(perfect):,} "
          f"from {perfect.id.nunique()} passages")
    rng = random.Random(SEED)
    picked = []
    for ds, n in [("SQuAD", N_SQUAD), ("HotpotQA", N_HOTPOT)]:
        ids = sorted(perfect[perfect.dataset == ds].id.unique())
        for pid in rng.sample(ids, n):
            options = perfect[perfect.id == pid].sort_values("source")
            picked.append(options.iloc[rng.randrange(len(options))])
    src = pd.DataFrame(picked).reset_index(drop=True)
    src.insert(0, "base_id", [f"B{i + 1}" for i in range(len(src))])
    return src


def main() -> None:
    src = select_base_items()
    rows = []
    for _, r in src.iterrows():
        for d in DIMS:
            q, change = PERTURBATIONS[r.base_id][d]
            iid = item_id(r.base_id, d)
            rows.append({"item_id": iid, "base_id": r.base_id, "dataset": r.dataset,
                         "valid_question_source_model": r.model, "qgeval_id": r.id,
                         "passage": r.passage, "answer": r.answer,
                         "reference_question": r.reference, "valid_question": r.prediction,
                         "target_dimension": d, "perturbation_rule": RULES[d],
                         "perturbed_question": q, "perturbation_type": change})
    ds = pd.DataFrame(rows)
    ds["only_target_broken"] = ONLY_TARGET_BROKEN
    ds["drafting_self_check"] = [("yes" if i in AUTHOR_CHANGED else lab)
                                 for i, lab in zip(ds.item_id, ds.only_target_broken)]
    ds["blind_main_problem"] = [BLIND_MAIN_PROBLEM.get(i, d) for i, d in zip(ds.item_id, ds.target_dimension)]

    # Quality checks
    assert len(ds) == 35 and ds.item_id.is_unique
    assert not ds.perturbed_question.duplicated().any()
    assert (ds.perturbed_question.str.lower() != ds.valid_question.str.lower()).all()
    counts = ds.groupby("target_dimension").only_target_broken.apply(lambda s: (s == "yes").sum()).reindex(DIMS)
    print("Isolation 'yes' per dimension:", counts.to_dict())
    print(f"Blind re-check: {(ds.blind_main_problem == ds.target_dimension).sum()} / {len(ds)} identified")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(OUT, index=False)
    print(f"Saved {OUT} ({len(ds)} items)")


if __name__ == "__main__":
    main()
