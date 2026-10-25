# Mem-RCA — Open-World Root Cause Analysis for Microservices

**Course:** AI253IA — Artificial Neural Networks & Deep Learning (PBL), RV College of Engineering
**Team:** Priyanshi Gupta (1RV24CI088) · Rishi Agarwal (1RV24CI099) · Sambhav Badaya (1RV24CI106)

Mem-RCA finds which microservice caused a failure, names the failure type — and says **NOVEL**
when the failure is of a type it has never seen, instead of guessing a wrong known type.
Recurring novel failures are clustered into candidate new types and enrolled into memory
without retraining.

## Folder structure

```
ANN PROJECT/
├── configs/data.yaml      dataset paths, fault types, open-world folds
├── src/memrca/            the Python package (data loading, models, evaluation)
├── scripts/               command-line entry points (download, explore, train, ...)
├── data/raw/              RCAEval dataset (not committed to git)
└── experiments/           figures, logs, results (not committed to git)
```

## Setup (Windows, Python 3.13)

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Check the GPU: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

## Dataset: RCAEval

Public benchmark (MIT licence) with 735 failure cases from three microservice systems
(Online Boutique `ob`, Sock Shop `ss`, Train Ticket `tt`).

| Suite | Cases | Data | Use in this project |
|---|---|---|---|
| **RE2** (start here) | 270 | metrics, logs, traces | main open-world experiments |
| RE1 | 375 | metrics | localisation, LSTM training |
| RE3 | 90 | code-level faults | extra NOVEL test set |

Each case is a folder `<suite><system>_<service>_<fault>_<repetition>`
(e.g. `re2ob_checkoutservice_cpu_1`) containing `metrics.parquet`, `logs.parquet`,
`traces.parquet` and `inject_time.txt` (data before this timestamp is normal, after it is faulty).
`cases.parquet` indexes all cases (root-cause service, fault type, …).

```
python scripts/download.py                 # RE2 into data/raw (about 2 GB)
python scripts/explore_case.py             # summary + figures for one case
python scripts/explore_case.py --case re2tt_ts-order-service_cpu_1
```

Second dataset (later, around week 5): PetShop — github.com/amazon-science/petshop-root-cause-analysis

## Week 1 plan

1. Download RE2 into `data/raw/`.
2. Explore cases: sampling interval, metric types per service, what each fault does.
3. Preprocess: normalise with normal-period statistics, cut windows, handle missing metrics.
4. Build service graphs from traces (Sock Shop: documented architecture if traces are missing).
5. Build the open-world folds (`configs/data.yaml`) and the held-out services.
