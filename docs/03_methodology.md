# Methodology — Dynamic Retention Benchmarking

Two halves, one API:
1. **Predict** the org's retention rate from its current state — hierarchical Ridge per sector.
2. **Recommend** the next HR action — ε-greedy contextual bandit keyed by sector.

---

## 1. EDA plan
- Per-sector retention distribution (median, IQR, sample size).
- Pairwise correlation of `comp_percentile`, `training_hours`, `manager_quality` with retention.
- Per-(sector, action) cell counts — flag arms with n < 30 (under-explored).

## 2. Featurization
Standardised numeric features: `headcount` (log-scaled), `comp_percentile`, `training_hours`, `manager_quality`. No categorical encoding because each Ridge fit is *within* a sector — sector is the partition, not a feature.

## 3. Per-sector Ridge

For each sector $s$, we fit independently

$$ \hat{r}_s(\mathbf{x}) = \mathbf{w}_s^\top \mathbf{x} + b_s, \qquad \mathbf{w}_s = \arg\min_{\mathbf{w}} \lVert X_s \mathbf{w} - \mathbf{y}_s\rVert_2^2 + \alpha \lVert\mathbf{w}\rVert_2^2 $$

with $\alpha = 1.0$. The **hierarchy is structural**: the per-sector models share a fitting recipe but never pool weights. A global Ridge would smear the regulated-vs-private gap (e.g. Public > Hospitality on retention by ~25 pts) — see §7 RMSE table.

A future Bayesian-hierarchical version would put a prior on $\mathbf{w}_s$ centred on a global $\bar{\mathbf{w}}$, partially pooling the small-n sectors.

## 4. ε-greedy contextual bandit

For each (sector $s$, action $a$) we maintain a running mean reward:

$$ \mu_{s,a} \leftarrow \mu_{s,a} + \frac{r - \mu_{s,a}}{n_{s,a} + 1}, \qquad n_{s,a} \leftarrow n_{s,a} + 1 $$

Action selection at decision time:

$$ a^\star = \begin{cases} \text{Uniform}(\mathcal{A}) & \text{w.p. } \varepsilon \\ \arg\max_{a \in \mathcal{A}} \mu_{s,a} & \text{otherwise} \end{cases} $$

with $\varepsilon = 0.10$. The dashboard surfaces the **top-3 actions ranked by $\mu_{s,a}$**, plus the n they were averaged over so the HR leader can judge confidence.

## 5. Cross-validation
- 5-fold on each sector independently for the Ridge models.
- Bandit evaluated by **replay** on the held-out year-2024 rows: simulate pulling the recommended arm and accumulate rewards vs random.

## 6. Evaluation metrics
| Model | Primary | Secondary |
|-------|---------|-----------|
| Per-sector Ridge | RMSE | MAE, R² |
| Bandit | cumulative reward vs random | per-arm coverage (n) |
| `/benchmark` | p95 latency | response payload size |

## 7. Why hierarchical (not pooled)
| Sector | Pooled Ridge MAE | Per-sector Ridge MAE |
|--------|------------------|----------------------|
| Public | 0.083 | 0.041 |
| Hospitality | 0.092 | 0.052 |
| ICT | 0.075 | 0.048 |
| All sectors avg | 0.085 | 0.054 |

(Targets, to be filled by `notebooks/04_eval`.)

## 8. References
- Hoerl & Kennard, *Ridge Regression: Biased Estimation for Nonorthogonal Problems*, 1970.
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. — §2 on ε-greedy.
- Li et al., *A Contextual-Bandit Approach to Personalized News Recommendation* (LinUCB), 2010.
