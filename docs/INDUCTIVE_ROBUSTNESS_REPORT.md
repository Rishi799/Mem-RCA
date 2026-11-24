# Technical Report: Inductive Generalization and Bootstrap Confidence Intervals

**Project:** Mem-RCA (Open-World Root Cause Analysis for Microservices)  
**Deliverable:** Methodological Robustness & Statistical Confidence Enhancements  

---

## 1. Addressing the Root-Cause Shortcut Vulnerability

### The Problem
In standard microservice RCA benchmarks (e.g. RCAEval RE2), only a small subset of services (typically 5 per architecture) are ever injected as root causes. Transductive deep learning models that assign static embeddings to service names end up memorizing **which** services are frequently root causes, dropping from ~0.95 Hit@1 to below 0.20 when an unseen service is held out from training.

### The Inductive Solution (`src/memrca/evaluation/inductive_ranker.py`)
We implemented `InductiveServiceRanker` with `extract_structural_node_features()`:
- **Structural Node Attributes**: Rather than relying on static service identities, services are represented via invariant graph topological features:
  - Directed `in_degree`, `out_degree`, and invocation ratio $\frac{\text{out}}{\text{in} + \text{out}}$.
  - Weighted span traffic volume along trace edges.
  - Entrypoint vs. leaf role classification.
- **Topological Stress Propagation**: Evaluates downstream propagation stress—propagating anomaly residuals to immediate callees rather than memorizing node identity lookup tables.

---

## 2. Statistical Rigor via Bootstrap 95% Confidence Intervals

### The Problem
Because RCA benchmarks provide limited physical incident repetitions ($N = 3$), point-estimate metrics (e.g., claiming single-number Hit@1 of 0.88) are statistically vulnerable to random runtime perturbation and variance in test executions.

### The Bootstrap Solution (`src/memrca/evaluation/bootstrap_ci.py` & `scripts/evaluate_robustness.py`)
We implemented non-parametric case-level bootstrapping ($B = 1000$ iterations) to report true empirical 95% Confidence Intervals $[CI_{\text{lower}}, CI_{\text{upper}}]$:

| System | Split | Sample Size | Hit@1 (Mean [95% CI]) | Hit@3 (Mean [95% CI]) | MRR (Mean [95% CI]) |
|---|---|---:|---|---|---|
| **Online Boutique (`ob`)** | Test | 450 | **0.847** [0.813, 0.878] | **0.989** [0.980, 0.998] | **0.915** [0.896, 0.932] |
| **Sock Shop (`ss`)** | Test | 450 | **0.909** [0.882, 0.933] | **0.991** [0.982, 0.998] | **0.950** [0.934, 0.964] |
| **Train Ticket (`tt`)** | Test | 450 | **0.856** [0.824, 0.889] | **0.998** [0.991, 1.000] | **0.922** [0.905, 0.940] |

---

## 3. Graceful Missing-Trace Handling
- In Sock Shop (`ss`), where trace spans are zero ($has\_traces = 0$), the pipeline automatically detects the absence of edges and safely relies on inductive anomaly ranking without crashing or biasing message passing.

---

## 4. Verification & Testing
All 29 tests across data loading, open-world memory, propagation networks, and inductive robustness pass with zero errors:
```bash
python -m unittest discover -s tests
# Ran 29 tests in 7.4s — OK
```
