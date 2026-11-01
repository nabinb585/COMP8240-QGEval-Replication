"""
NewsQA: human annotation vs LLM judge vs automatic metrics.

Inputs (from notebook 02 and data/newsqa/):
  newsqa_annotation_sheet.xlsx  - blind sheet; the author's scores for the 15 items below
  newsqa_annotation_key.csv     - hidden key: item -> source system and instance id
  newsqa_qgeval_format.csv      - NewsQA sample with generated questions and metric scores
  data/newsqa/newsqa_llm_judge_scores.csv - LLM-judge scores for all 45 items

Outputs: per-system mean scores, human-vs-judge agreement per dimension, and
Pearson correlations between automatic metrics and judge scores (30 generated questions).

Run from the repository root:
    python src/newsqa_agreement.py --sheet newsqa_annotation_sheet.xlsx \
        --key newsqa_annotation_key.csv --data newsqa_qgeval_format.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DIMS = ["fluency", "clarity", "conciseness", "relevance",
        "consistency", "answerability", "answer_consistency"]
METRICS = ["BLEU-4", "ROUGE-L", "METEOR", "BERTScore"]
HUMAN_ITEMS = ["A04", "A05", "A07", "A08", "A10", "A15", "A16", "A17",
               "A18", "A20", "A27", "A29", "A34", "A40", "A43"]   # 5 passages x 3 sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="newsqa_annotation_sheet.xlsx")
    ap.add_argument("--key", default="newsqa_annotation_key.csv")
    ap.add_argument("--data", default="newsqa_qgeval_format.csv")
    ap.add_argument("--judge", default="data/newsqa/newsqa_llm_judge_scores.csv")
    args = ap.parse_args()

    key = pd.read_csv(args.key)
    data = pd.read_csv(args.data)
    llm = pd.read_csv(args.judge).merge(key, on="item")

    human = pd.read_excel(args.sheet, sheet_name="annotate")
    human[DIMS] = human[DIMS].apply(pd.to_numeric, errors="coerce")
    human = human[human.item.isin(HUMAN_ITEMS)].merge(key, on="item")
    assert human[DIMS].notna().all().all(), "All 15 human rows need 7 scores"

    print("Human mean per system (5 questions each):")
    print(human.groupby("source")[DIMS].mean().round(2).to_string())
    print("\nLLM-judge mean per system (15 questions each):")
    print(llm.groupby("source")[DIMS].mean().round(2).to_string())

    both = human.merge(llm[["item"] + DIMS], on="item", suffixes=("_h", "_llm"))
    rows = []
    for d in DIMS:
        h, j = both[f"{d}_h"], both[f"{d}_llm"]
        rho = spearmanr(h, j)[0] if h.nunique() > 1 and j.nunique() > 1 else np.nan
        rows.append({"dimension": d, "exact agreement %": round((h == j).mean() * 100, 1),
                     "within 1 point %": round(((h - j).abs() <= 1).mean() * 100, 1),
                     "Spearman": round(rho, 3)})
    agree = pd.DataFrame(rows)
    hv = both[[f"{d}_h" for d in DIMS]].values
    jv = both[[f"{d}_llm" for d in DIMS]].values
    print("\nHuman vs LLM judge (same 15 questions):")
    print(agree.to_string(index=False))
    print(f"Overall exact agreement: {(hv == jv).mean() * 100:.1f}%")

    gen = llm[llm.source != "reference"].copy()
    for i, r in gen.iterrows():
        row = data[data.id == r.instance].iloc[0]
        for m in METRICS:
            gen.loc[i, m] = row[f"{m}_{r.source}"]
    corr = pd.DataFrame({m: [pearsonr(gen[d], gen[m])[0] if gen[d].nunique() > 1 else np.nan
                             for d in DIMS] for m in METRICS}, index=DIMS).T.round(3)
    print(f"\nPearson r, metric vs LLM-judge score (n = {len(gen)} generated questions):")
    print(corr.to_string())


if __name__ == "__main__":
    main()
