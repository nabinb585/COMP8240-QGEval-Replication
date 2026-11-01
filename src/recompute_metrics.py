"""
COMP8240 Final Project - recompute reference-based metrics from raw text

replicate_table5.py reuses the authors' saved metric scores. This script goes one
step further: it recomputes metric scores from the raw questions using the same
method as the authors' code (metrics/base_evaluator.py), checks they match the
released scores, and recomputes the correlations with human judgements.

Run from inside the cloned QGEval folder (see README):
    pip install pandas openpyxl scipy nltk rouge-score bert-score
    python ../src/recompute_metrics.py

BLEU-4 always runs. ROUGE-L, METEOR and BERTScore run if their libraries are
installed (they install fine on Google Colab). BERTScore is slow without a GPU.
"""

import sys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.insert(0, "metrics")
from coeff import Coeff

DIMS = ["fluency", "clarity", "conciseness", "relevance",
        "consistency", "answerability", "answer_consistency"]
SHORT = ["Flu", "Clar", "Conc", "Rel", "Cons", "Ans", "AnsC"]

human = pd.read_excel("data/scores.xlsx")
released = pd.read_excel("data/metric_result.xlsx")
hypos = human["prediction"].astype(str).tolist()
refs = human["reference"].astype(str).tolist()
print(f"Recomputing metrics for {len(hypos)} generated questions\n")

mine = {}

# --- BLEU-4: sentence BLEU with smoothing method1, whitespace tokens (as in authors' code)
smooth = SmoothingFunction().method1
mine["BLEU-4"] = [round(sentence_bleu([r.strip().split(" ")], h.strip().split(" "),
                                      smoothing_function=smooth), 4)
                  for h, r in zip(hypos, refs)]

# --- ROUGE-L F1 with stemming
try:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    mine["ROUGE-L"] = [round(sc.score(h, r)["rougeL"].fmeasure, 4) for h, r in zip(hypos, refs)]
except ImportError:
    print("ROUGE-L skipped (pip install rouge-score)")

# --- METEOR (needs WordNet)
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    mine["METEOR"] = [round(meteor_score([r.strip().split(" ")], h.strip().split(" ")), 4)
                      for h, r in zip(hypos, refs)]
except Exception as e:
    print(f"METEOR skipped ({type(e).__name__})")

# --- BERTScore F1 (default English model, as in authors' code)
try:
    from bert_score import score as bert_score
    _, _, f1 = bert_score(hypos, refs, lang="en", verbose=False)
    mine["BERTScore"] = [round(x, 4) for x in f1.tolist()]
except Exception as err:  # bert-score missing or model download unavailable
    print(f"BERTScore skipped ({type(err).__name__})")

# --- Compare with released scores, then correlate with human judgements
coeff = Coeff()
summary, corr = [], {}
for m, vals in mine.items():
    diff = (pd.Series(vals) - released[m]).abs()
    summary.append({"metric": m,
                    "identical_scores": f"{int((diff <= 0.0001).sum())}/{len(vals)}",
                    "max_abs_diff": round(diff.max(), 4)})
    corr[m] = [coeff.apply(human[d].tolist(), vals)[0] for d in DIMS]

print("Recomputed scores vs authors' released scores:")
print(pd.DataFrame(summary).set_index("metric").to_string())

print("\nPearson correlation with human scores, using MY recomputed scores:")
print(pd.DataFrame(corr, index=SHORT).T.to_string())

out = human[["id", "source", "prediction", "reference"]].copy()
for m, vals in mine.items():
    out[f"{m}_recomputed"] = vals
out.to_csv("../results/recomputed_metric_scores.csv", index=False)
print("\nSaved: ../results/recomputed_metric_scores.csv")
