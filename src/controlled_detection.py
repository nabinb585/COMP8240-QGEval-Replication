"""
Detection test on the Controlled Question Quality Dataset.

For each item, every metric scores the valid question and the perturbed question
against the QGEval reference question. An error is "detected" when the perturbed
question scores lower than the valid one. Reports the detection rate (%) per
target dimension.

BERTScore downloads roberta-large from Hugging Face; if it is unavailable the
script reports the other three metrics only.

Run from the repository root:
    python src/controlled_detection.py
Output: results/controlled_detection_rates.csv
"""

from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

DATA = Path("data/controlled/controlled_question_quality_dataset.csv")
OUT = Path("results/controlled_detection_rates.csv")
DIMS = ["fluency", "clarity", "conciseness", "relevance",
        "consistency", "answerability", "answer_consistency"]

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def metric_scores(hyps, refs):
    # Whitespace tokenisation, as in QGEval's metric code.
    out = {
        "BLEU-4": [sentence_bleu([r.split(" ")], h.split(" "), smoothing_function=smooth) for h, r in zip(hyps, refs)],
        "ROUGE-L": [rouge.score(h, r)["rougeL"].fmeasure for h, r in zip(hyps, refs)],
        "METEOR": [meteor_score([r.split(" ")], h.split(" ")) for h, r in zip(hyps, refs)],
    }
    try:
        from bert_score import score as bert_score
        out["BERTScore"] = bert_score(hyps, refs, lang="en", verbose=False)[2].tolist()
    except Exception as err:  # model download unavailable offline
        print(f"BERTScore skipped ({type(err).__name__})")
    return pd.DataFrame(out)


def main() -> None:
    ds = pd.read_csv(DATA)
    refs = ds.reference_question.tolist()
    valid = metric_scores(ds.valid_question.tolist(), refs)
    pert = metric_scores(ds.perturbed_question.tolist(), refs)
    metrics = list(valid.columns)
    for m in metrics:
        ds[f"{m}_detected"] = pert[m].values < valid[m].values

    rates = pd.DataFrame({m: ds.groupby("target_dimension")[f"{m}_detected"].mean().reindex(DIMS) * 100
                          for m in metrics}).round(0)
    rates.loc["ALL"] = [round(ds[f"{m}_detected"].mean() * 100) for m in metrics]
    print("Detection rate (%): perturbed question scored lower than the valid one\n")
    print(rates.to_string())

    w = lambda s: s.str.split().str.len()
    ds["length_change_words"] = w(ds.perturbed_question) - w(ds.valid_question)
    print("\nMean length change (words) per target dimension:")
    print(ds.groupby("target_dimension").length_change_words.mean().reindex(DIMS).round(1).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(OUT)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
