"""
COMP8240 Final Project - Replication of QGEval (Fu et al., EMNLP 2024)

Reproduces:
  - Table 5: Pearson correlation between 15 automatic metrics and human scores
             on 7 dimensions (3,000 questions)
  - Table 6: Correlation of LLM-based evaluators with answerability (450 questions)

Uses the authors' released files and their own correlation class (metrics/coeff.py).

Run from inside the cloned QGEval folder (see README):
    pip install pandas openpyxl scipy
    python ../src/replicate_table5.py
"""

import ast
import sys
import pandas as pd

sys.path.insert(0, "metrics")
from coeff import Coeff          # authors' implementation of Pearson/Spearman/Kendall

DIMS = ["fluency", "clarity", "conciseness", "relevance",
        "consistency", "answerability", "answer_consistency"]
SHORT = ["Flu", "Clar", "Conc", "Rel", "Cons", "Ans", "AnsC"]

METRICS = ["BLEU-4", "ROUGE-L", "METEOR", "BERTScore", "MoverScore", "BLEURT",
           "BARTScore-ref", "GPTScore-ref", "Q-BLEU4", "QSTS",            # reference-based
           "BARTScore-src", "GPTScore-src", "QRelScore", "UniEval", "RQUGE"]  # reference-free

# Values reported in Table 5 of the paper (for comparison)
PAPER_T5 = {
    "BLEU-4":        [0.028, 0.049, 0.138, 0.041, 0.032, 0.080, 0.162],
    "ROUGE-L":       [0.080, 0.086, 0.234, 0.085, 0.079, 0.127, 0.233],
    "METEOR":        [0.020, 0.088, 0.106, 0.079, 0.059, 0.131, 0.253],
    "BERTScore":     [0.140, 0.123, 0.313, 0.113, 0.091, 0.131, 0.231],
    "MoverScore":    [0.070, 0.075, 0.209, 0.071, 0.058, 0.101, 0.188],
    "BLEURT":        [0.078, 0.105, 0.179, 0.104, 0.098, 0.144, 0.271],
    "BARTScore-ref": [0.087, 0.079, 0.235, 0.109, 0.078, 0.092, 0.190],
    "GPTScore-ref":  [0.069, 0.086, 0.182, 0.006, 0.054, 0.106, 0.187],
    "Q-BLEU4":       [0.072, 0.082, 0.216, 0.058, 0.075, 0.113, 0.198],
    "QSTS":          [0.016, 0.104, 0.015, 0.077, 0.043, 0.130, 0.250],
    "BARTScore-src": [-0.148, -0.035, -0.511, 0.053, -0.001, 0.018, -0.015],
    "GPTScore-src":  [0.134, 0.104, -0.052, 0.416, 0.197, 0.148, 0.236],
    "QRelScore":     [-0.213, -0.096, -0.553, 0.032, 0.002, -0.026, -0.025],
    "UniEval":       [0.370, 0.219, 0.259, 0.153, 0.156, 0.207, 0.356],
    "RQUGE":         [0.045, 0.092, 0.126, 0.070, 0.200, 0.211, 0.561],
}
PAPER_T6 = {"GPTScore-src": 0.187, "UniEval": 0.215, "RQUGE": 0.250,
            "GPT3.5": 0.195, "G-EVAL-gpt3.5": 0.228,
            "GPT4": 0.296, "G-EVAL-gpt4": 0.356}

coeff = Coeff()
pd.set_option("display.width", 200)

# ---------------------------------------------------------------- 1. Load data
human = pd.read_excel("data/scores.xlsx")          # 3,000 questions + 7 human scores
metric = pd.read_excel("data/metric_result.xlsx")  # 3,000 questions + 15 metric scores
print(f"Human scores : {human.shape[0]} questions, {len(DIMS)} dimensions")
print(f"Metric scores: {metric.shape[0]} questions, {len(METRICS)} metrics")

# Make sure both files describe the same questions in the same order
key = ["id", "source", "prediction"]
assert (human[key].astype(str).values == metric[key].astype(str).values).all(), \
    "Rows of scores.xlsx and metric_result.xlsx do not line up"
df = pd.concat([human, metric[METRICS]], axis=1)
print("Rows aligned: OK\n")

# Multi-dimensional metrics (GPTScore, UniEval) store one score per dimension as
# a dictionary, e.g. "{'fluency': 0.91, 'clarity': 0.84, ...}". For these, each
# human dimension is compared with the metric's matching sub-score.
MULTI_DIM = [m for m in METRICS if df[m].dtype == object or str(df[m].dtype) == "str"]
for m in MULTI_DIM:
    df[m] = df[m].apply(ast.literal_eval)
print("Multi-dimensional metrics (scored per dimension):", MULTI_DIM, "\n")

def metric_scores(frame, m, dim):
    if isinstance(frame[m].iloc[0], dict):
        return frame[m].apply(lambda s: s[dim]).tolist()
    return frame[m].tolist()

# ---------------------------------------------------- 2. Reproduce Table 5
mine, diffs = {}, {}
for m in METRICS:
    mine[m] = [coeff.apply(df[d].tolist(), metric_scores(df, m, d))[0] for d in DIMS]  # Pearson
    diffs[m] = [round(a - b, 3) for a, b in zip(mine[m], PAPER_T5[m])]

t_mine = pd.DataFrame(mine, index=SHORT).T
t_mine["Avg"] = t_mine.mean(axis=1).round(3)
t_paper = pd.DataFrame(PAPER_T5, index=SHORT).T
max_diff = pd.DataFrame(diffs, index=SHORT).T.abs().max().max()
n_match = int((pd.DataFrame(diffs, index=SHORT).T.abs() <= 0.001).sum().sum())

print("=" * 78)
print("TABLE 5 (reproduced): Pearson correlation, metric vs human, 3,000 questions")
print("=" * 78)
print(t_mine.to_string())
print(f"\nCells matching the paper (within 0.001): {n_match} / {t_paper.size}")
print(f"Largest absolute difference from paper : {max_diff:.3f}")

print("\nBest metric per dimension (reproduced):")
for s in SHORT:
    print(f"  {s:5s} -> {t_mine[s].idxmax():14s} r = {t_mine[s].max():.3f}")

# ---------------------------------------------------- 3. Reproduce Table 6
ans = pd.read_excel("data/test_answerability.xlsx")   # 450 questions, 30 passages
print("\n" + "=" * 78)
print(f"TABLE 6 (reproduced): evaluators vs human answerability, {len(ans)} questions")
print("=" * 78)
for col in PAPER_T6:
    if isinstance(ans[col].iloc[0], str) and ans[col].iloc[0].startswith("{"):
        ans[col] = ans[col].apply(ast.literal_eval)
t6 = []
for col, paper_val in PAPER_T6.items():
    r = coeff.apply(ans["answerability"].tolist(), metric_scores(ans, col, "answerability"))[0]
    t6.append({"evaluator": col, "reproduced": r, "paper": paper_val,
               "match": "yes" if abs(r - paper_val) <= 0.001 else "NO"})
t6 = pd.DataFrame(t6).set_index("evaluator")
print(t6.to_string())

# ---------------------------------------------------- 4. Save outputs
with pd.ExcelWriter("../results/replication_results.xlsx") as xw:
    t_mine.to_excel(xw, sheet_name="Table5_reproduced")
    t_paper.to_excel(xw, sheet_name="Table5_paper")
    pd.DataFrame(diffs, index=SHORT).T.to_excel(xw, sheet_name="Table5_difference")
    t6.to_excel(xw, sheet_name="Table6_answerability")
print("\nSaved: ../results/replication_results.xlsx")
