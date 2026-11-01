# Does QGEval Generalise? — COMP8240 Final Project

**Nabin Basnet (48441147) · COMP8240 Applications of Data Science · Macquarie University · Session 2, 2026**

This repository replicates **QGEval** (Fu et al., EMNLP 2024), a benchmark that scores generated questions on seven quality dimensions (fluency, clarity, conciseness, relevance, consistency, answerability, answer consistency), and tests whether its main finding — automatic metrics agree only weakly with human judgement — holds on new data:

1. **Replication** of QGEval Table 5 (15 metrics × 7 dimensions, 3,000 questions) and Table 6 (LLM evaluators), plus recomputation of four metrics from raw text.
2. **New existing data:** a NewsQA sample (CNN news), two T5 question generators, blind annotation and an LLM judge.
3. **New constructed data:** the *Controlled Question Quality Dataset* — 35 items in which exactly one dimension is deliberately broken — used to test which errors metrics detect.

QGEval paper: https://aclanthology.org/2024.emnlp-main.658/ · QGEval code and data: https://github.com/WeipingFu/QGEval

## Repository layout

```
notebooks/
  01_replication_qgeval.ipynb     Replication of Tables 5 and 6 + metric recomputation (Colab)
  02_newsqa_new_data.ipynb        NewsQA collection, preprocessing, QG, metrics, annotation sheet (Colab, outputs saved)
  03_controlled_dataset.ipynb     Controlled dataset: selection, checks, detection test, blind re-check (Colab, outputs saved)
src/
  replicate_table5.py             Level 1: Table 5 and Table 6 from released scores, authors' correlation code
  recompute_metrics.py            Level 2: BLEU-4, ROUGE-L, METEOR, BERTScore recomputed from raw text
  build_controlled_dataset.py     Rebuilds the 35-item controlled dataset (seed 42) with its labels
  controlled_detection.py         Detection rate of each metric per target dimension
  newsqa_agreement.py             Human vs LLM judge agreement and metric correlations on NewsQA
data/
  controlled/controlled_question_quality_dataset.csv   35 items (schema below)
  newsqa/newsqa_llm_judge_scores.csv                   LLM-judge scores for the 45 blind NewsQA questions
results/                          Outputs reported in the paper
requirements.txt
```

`QGEval/` (the authors' repository) is cloned at run time and is not redistributed here, because it declares no licence.

## How to run

**Option A — Google Colab (recommended, matches the paper).** Open each notebook in Colab, set *Runtime → Change runtime type → T4 GPU*, and run all cells top to bottom. Notebook 01 clones this repository and QGEval itself. A GPU is needed only for BERTScore and for question generation in notebook 02.

**Option B — local (Python 3.10+).**

```bash
git clone https://github.com/nabinb585/COMP8240-QGEval-Replication.git
cd COMP8240-QGEval-Replication
pip install -r requirements.txt
git clone https://github.com/WeipingFu/QGEval.git

# Replication (run inside QGEval/, outputs go to results/)
cd QGEval
python ../src/replicate_table5.py      # ~10 s, CPU only
python ../src/recompute_metrics.py     # BERTScore downloads roberta-large; skipped if offline
cd ..

# Controlled dataset
python src/build_controlled_dataset.py # rebuilds data/controlled/...csv
python src/controlled_detection.py     # detection rates -> results/
```

NewsQA (notebook 02) downloads `lucadiliello/newsqa` and two `lmqg` T5 checkpoints from Hugging Face. Every random step uses seed 42.

## Main results

| Experiment | Result |
|---|---|
| Table 5 replication (105 correlations) | 104 within ±0.001 of the paper; one sign flip at magnitude 0.001 |
| Table 6 replication (7 evaluators) | 7 / 7 identical |
| Recomputed metrics (3,000 questions) | BLEU-4 2,998, METEOR 2,998, ROUGE-L 2,994 identical scores; BERTScore correlations within 0.008 |
| Second environment (Linux CPU, no GPU) | Level 1 and BLEU-4/ROUGE-L/METEOR reproduced identically |
| NewsQA metric means | BLEU-4/ROUGE-L/METEOR drop sharply vs SQuAD/HotpotQA and rank T5-base above T5-large, opposite to human and judge ratings |
| NewsQA human vs LLM judge (15 questions) | 85.7% exact agreement, 100% within one point; clarity lowest (66.7%) |
| Controlled detection (35 items) | Conciseness errors detected 100%; changed fact 60–80%; wrong target answer 60% for all four metrics |

Full tables are in `results/` and in the notebooks' saved outputs.

## Controlled dataset schema

| Column | Meaning |
|---|---|
| `item_id` | Base item and target dimension, e.g. `B1-FLUE`, `B4-ANSC` |
| `dataset`, `qgeval_id`, `valid_question_source_model` | Origin of the base question in QGEval |
| `passage`, `answer`, `reference_question` | QGEval inputs and human reference |
| `valid_question` | QGEval-generated question rated 3/3 on all dimensions by all three annotators |
| `target_dimension`, `perturbation_rule` | The one dimension the item breaks, and the rule used |
| `perturbed_question`, `perturbation_type` | The broken question and what was changed |
| `only_target_broken` | Author's label: `yes` (only the target broken) or `partly` (another dimension also affected) |
| `drafting_self_check` | Isolation check recorded when the item was drafted (differs from the author's label on 2 items) |
| `blind_main_problem` | Main error named by the author with the target hidden (matches the target on 33 / 35) |

Perturbations were drafted with a large language model under the fixed rules and reviewed item by item by the author; no question text needed rewriting.

## Data sources and licences

- **QGEval** data and metric code: https://github.com/WeipingFu/QGEval (no licence declared; cloned at run time, not redistributed).
- **SQuAD v1.1** and **HotpotQA**: CC BY-SA 4.0. The controlled dataset reuses their passages under the same licence.
- **NewsQA** (MRQA 2019 version, `lucadiliello/newsqa`): code MIT; CNN articles are used by permission and CNN retains all rights. This repository does not redistribute the articles; notebook 02 downloads them at run time, and the data files here hold only item identifiers, scores and short notes.

## Environment used for the reported run

Google Colab, Tesla T4 GPU, Python 3.13.15, pandas 2.2.3, SciPy 1.16.3, PyTorch 2.11.0 (CUDA 13.0), transformers 5.17.0, datasets 4.8.5.
