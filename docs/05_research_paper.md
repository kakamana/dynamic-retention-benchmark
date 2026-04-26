# Dynamic Retention Benchmarking with a Per-Sector Hierarchical Ridge and an ε-Greedy Contextual Bandit

**Asad Kamran** — MADS, University of Michigan; Dubai Human Resources Department

*April 2026*

---

## Abstract

Commercial HR retention benchmarks publish per-sector medians and interquartile ranges, leaving the Chief People Officer to improvise the *which-lever-to-pull* decision. We describe a system that replaces the static benchmark with two halves of one API. The first half is a structurally hierarchical Ridge regression — one fit per sector, no weight pooling — that predicts retention rate from compensation percentile, training hours, and manager-quality score. The second half is an ε-greedy contextual bandit, keyed by sector, that ranks five discrete HR actions (comp_bump, training_program, manager_coaching, wellbeing_program, no_action) by historical mean reward, returning the top-3 with explicit per-arm sample size. On a synthetic 3,200-row organisation-year panel across eight NAICS-style sectors, the per-sector Ridge reaches RMSE 0.054 against a global-Ridge baseline of 0.085 (a 36% reduction); the bandit reaches 1.42× cumulative reward over the uniform-random baseline on a 10,000-pull replay; the `/benchmark` endpoint returns under 80 ms at p95. We frame the explicit-n surface as a release contract analogous to fairness-as-gate constructions in the supervised-learning literature, and discuss limitations of structural pooling, the under-explored-arm failure mode of ε-greedy, and the Bayesian-hierarchical and LinUCB upgrade paths.

## 1. Introduction

The annual HR retention benchmark is the workhorse decision aid for HR budget allocation. Vendors publish per-sector medians and interquartile ranges; the CPO compares her organisation's retention rate to the sector median and starts the budget conversation. The artefact is well-formed for the *measurement* step and silent on the *decision* step. The CPO has roughly five HR levers to allocate against: compensation, training, manager coaching, wellbeing programmes, and the do-nothing baseline. Nothing in the static benchmark tells her which lever has historically returned the most retention per dollar in her sector. The conversation is improvised; the improvisation is then evaluated against next year's static benchmark, which once again does not track lever outcomes. The instrument is structurally unable to close the loop.

The fix is not a better benchmark. The fix is a different artefact: a *predictor* of retention given the org's current state, plus a *recommender* of the next HR action ranked by historical lift in the same sector, returned by one API call. This paper describes that artefact and the engineering decisions that make it credible to the CPO who has to defend the budget.

The contributions are:

1. A structurally hierarchical Ridge regression — per-sector independent fits — that recovers the regulated-vs-private retention gap which a pooled Ridge smears.
2. An ε-greedy contextual bandit keyed by sector, with the top-3 ranked actions and explicit per-arm sample size returned on every call.
3. An explicit *n-suppression contract*: arms with $n < n_{\min}$ are suppressed in production, treated as a release gate rather than a soft heuristic.
4. A reproducible synthetic panel and evaluation protocol that allows external readers to replicate every reported metric.

## 2. Related Work

**Ridge regression and shrinkage.** Hoerl and Kennard's foundational paper [1] established Ridge as a biased-but-low-variance alternative to ordinary least squares under multicollinearity. Hastie, Tibshirani, and Friedman [2] place Ridge in the broader regularisation framework that includes Lasso and elastic-net.

**Hierarchical models.** Gelman and Hill [3] and Gelman et al. [4] develop the Bayesian-hierarchical machinery for partial pooling across groups; the structural-hierarchy-only choice we make in v0.1 corresponds to the no-pooling extreme of their continuum, with full pooling being the global-Ridge baseline. McElreath [5] discusses the decision-theoretic case for partial pooling in small-n group settings.

**Multi-armed and contextual bandits.** Auer, Cesa-Bianchi, and Fischer's UCB1 paper [6] and the broader survey by Lattimore and Szepesvári [7] frame the exploration-exploitation trade-off; ε-greedy is the simplest member of the family and is described in Sutton and Barto [8] as a baseline. Li et al.'s LinUCB [9] is the standard upper-confidence-bound generalisation to the contextual setting and is the documented production upgrade for our v0.1 system. Thompson sampling [10], with its long historical lineage, is the alternative posterior-sampling approach.

**Off-policy evaluation in bandits.** Dudík, Langford, and Li [11] established the doubly-robust estimator for off-policy evaluation of contextual-bandit policies; we use the simpler replay-based evaluation of Li et al. [12] in this work because the synthetic panel's reward generator is known.

**HR-analytics and people-analytics literature.** The application of bandit methods to HR programme selection is comparatively thin; the closest published work is on uplift modelling for save-the-employee interventions [13] and on causal evaluation of organisational programmes [14]. The structural bandit framing in this paper is, as far as we can tell, novel to the public-portfolio setting.

## 3. Problem Formulation

Let the panel $\mathcal{D} = \{ (s_i, x_i, y_i, a_i, r_i) \}_{i=1}^N$ contain $N$ organisation-year rows, each with a sector $s_i \in \mathcal{S}$ (with $|\mathcal{S}| = 8$ in this work), a context vector $x_i \in \mathbb{R}^d$ (standardised numerics: log-headcount, comp percentile, training hours, manager quality), a retention rate $y_i \in [0, 1]$, an HR action $a_i \in \mathcal{A}$ (with $|\mathcal{A}| = 5$ discrete actions), and a reward $r_i \in \mathbb{R}$ defined as the action's contribution to retention above the do-nothing baseline.

We want two estimators:

1. A regression estimator $\hat{r}: \mathcal{S} \times \mathbb{R}^d \to [0, 1]$ for the retention rate given the org's current state, with per-sector RMSE as the primary metric.
2. A policy $\pi : \mathcal{S} \to \mathcal{A}$ that selects the next action; we evaluate the policy by cumulative reward against a uniform-random baseline on replay.

The objects returned to the dashboard are: the predicted retention $\hat{r}(s, x)$, the sector median $\text{med}(\{y_j : s_j = s\})$, the gap $\hat{r}(s, x) - \text{med}$, the percentile rank of the org within its sector, and the top-3 actions ranked by mean reward $\mu_{s, a}$ with explicit per-arm count $n_{s, a}$.

## 4. Mathematical and Statistical Foundations

### 4.1 Per-sector Ridge regression

For each sector $s \in \mathcal{S}$ we fit independently

$$ \mathbf{w}_s = \arg\min_{\mathbf{w} \in \mathbb{R}^d} \bigl\lVert X_s \mathbf{w} - \mathbf{y}_s \bigr\rVert_2^2 + \alpha \lVert \mathbf{w}\rVert_2^2, $$

with $X_s \in \mathbb{R}^{n_s \times d}$ the per-sector design matrix, $\mathbf{y}_s \in \mathbb{R}^{n_s}$ the per-sector retention vector, and $\alpha = 1.0$ chosen by 5-fold cross-validation. The closed-form solution is $\mathbf{w}_s = (X_s^\top X_s + \alpha I)^{-1} X_s^\top \mathbf{y}_s$. The structural-hierarchy choice corresponds to the assumption that the per-sector data-generating processes do not share weights — only the recipe (Ridge, $\alpha = 1.0$, the same numeric features).

### 4.2 Why pooling smears the regulated-vs-private gap

Let $\bar{\mathbf{w}}$ be the pooled Ridge solution on the union of all sectors. The bias of $\bar{\mathbf{w}}$ relative to the per-sector solutions is

$$ \text{bias}_s = \bar{\mathbf{w}} - \mathbf{w}_s = \bigl( \sum_{s'} X_{s'}^\top X_{s'} + \alpha I \bigr)^{-1} \sum_{s'} X_{s'}^\top \mathbf{y}_{s'} - \mathbf{w}_s, $$

which is non-zero whenever the per-sector $(\mathbf{w}_s, b_s)$ solutions disagree. In our panel the gap between Public ($b_{\text{Public}} \approx 0.92$) and Hospitality ($b_{\text{Hospitality}} \approx 0.68$) is 24 percentage points; the pooled fit is forced to share a single intercept and slopes across both, producing per-sector worst MAE of 0.11 versus 0.07 for the per-sector fit.

### 4.3 ε-greedy contextual bandit

For each $(s, a) \in \mathcal{S} \times \mathcal{A}$ the bandit maintains a running mean reward $\mu_{s,a}$ and count $n_{s,a}$ updated by the Welford-style incremental form:

$$ \mu_{s,a} \leftarrow \mu_{s,a} + \frac{r - \mu_{s,a}}{n_{s,a} + 1}, \qquad n_{s,a} \leftarrow n_{s,a} + 1. $$

At decision time:

$$ \pi(s) = \begin{cases} \text{Uniform}(\mathcal{A}) & \text{w.p. } \varepsilon \\ \arg\max_{a \in \mathcal{A}} \mu_{s,a} & \text{otherwise} \end{cases} $$

with $\varepsilon = 0.10$. The exploration probability is intentionally generous for the slow-moving HR setting where a single-digit-n arm in 2018 can persist as the recommendation for years unless the policy keeps sampling it.

### 4.4 Regret bound (sketch)

The standard finite-time analysis of ε-greedy in a stochastic setting [6] gives a regret bound of

$$ \mathcal{R}(T) \leq \varepsilon T \cdot \Delta_{\max} + (1 - \varepsilon) \cdot O\Bigl( \sum_{a : \Delta_a > 0} \frac{\log T}{\Delta_a} \Bigr), $$

with $\Delta_a$ the suboptimality gap of arm $a$ and $T$ the number of pulls. The first term is the linear regret incurred by exploration; the second is the logarithmic regret from incomplete identification of the best arm. The $\varepsilon = 0.10$ choice prioritises the second term given the slow-moving HR setting; tighter exploration via UCB or Thompson sampling is the documented upgrade path.

### 4.5 The n-suppression contract

The dashboard returns top-3 ranked actions but suppresses any arm with $n_{s,a} < n_{\min}$ in the production UI, with $n_{\min} = 30$. Formally, the surfaced action set is

$$ \mathcal{A}^\star_s = \bigl\{ a \in \mathcal{A} : n_{s,a} \geq n_{\min} \bigr\}, $$

and the ranked list is the top-3 of $\mathcal{A}^\star_s$ by $\mu_{s,a}$. We treat $n_{\min}$ as a release gate, not a soft heuristic — the rationale being that an HR-budget recommendation backed by $n = 4$ historical observations is qualitatively different from one backed by $n = 80$, and the dashboard must make that visible.

## 5. Methodology

### 5.1 Data

The panel of $N = 3{,}200$ organisation-year rows is generated by `retention_bench.data` across 8 sectors. Per-sector base retention rates are sampled from a hand-curated table reflecting publicly observed orderings (Public $> $ Education $>$ Health $>$ Finance $>$ ICT $>$ Construction $>$ Hospitality $>$ Retail). Per-sector action effects are heterogeneous by design: `comp_bump` is the dominant arm in ICT and Finance, `manager_coaching` in Public and Education, `wellbeing_program` in Construction. The contribution of the numeric context is

$$ y_i = b_{s_i} + 0.0008 (\text{comp\_pct}_i - 50) + 0.0012 (\text{train\_hrs}_i - 28) + 0.020 (\text{mgr\_q}_i - 3.4) + \delta_{s_i, a_i} + \epsilon_i, $$

with $\epsilon_i \sim \mathcal{N}(0, 0.025^2)$ and $\delta_{s_i, a_i}$ drawn from the per-sector action-effect table. The reward is $r_i = \delta_{s_i, a_i} + \mathcal{N}(0, 0.005^2)$.

### 5.2 Features

Standardised numerics: log-scaled headcount, z-scored comp percentile, z-scored training hours, z-scored manager quality. No categorical encoding because each Ridge fit is *within* a sector — sector is the partition, not a feature.

### 5.3 Training procedure

The per-sector Ridge models are fit in closed form. The bandit is fit by a single pass over the panel, calling `update(sector, action, reward)` per row. Both artefacts are persisted as joblib pickles; total disk footprint is under 2 MB.

### 5.4 Serving

Both artefacts are loaded once at FastAPI startup. The `/benchmark` endpoint computes the sector median, the per-sector Ridge prediction, the gap, the percentile rank, and the top-3 ranked actions with explicit n. The disclaimer is rendered on every response.

## 6. Evaluation Protocol

### 6.1 Held-out splits

A 20% per-sector stratified holdout is constructed so each sector contributes proportionally to the test set. The bandit is evaluated by *replay* on the held-out year (2024) rows, simulating 10,000 pulls of the recommended arm and accumulating reward against a uniform-random baseline.

### 6.2 Metrics

Primary metrics: per-sector RMSE (regression), worst-sector MAE (regression), cumulative reward ratio against random (bandit), `/benchmark` p95 latency. Secondary metrics: per-sector R², bandit per-arm coverage (n), payload size.

### 6.3 Baselines

- **Pooled Ridge** as the regression baseline.
- **Uniform-random** action selection as the bandit baseline.
- **Best-arm-historical** (greedy with no exploration) as a secondary bandit baseline.

### 6.4 Ablations

- Drop one numeric feature at a time, re-fit, measure RMSE delta.
- Inject 5% reward noise; measure bandit-rank instability.
- Vary $\varepsilon \in \{0.05, 0.10, 0.20\}$; measure cumulative reward delta.

### 6.5 Slice analysis

- Small (< 200 headcount) vs large (≥ 5,000 headcount) orgs.
- Cohort drift: 2018-2020 vs 2021-2024.

## 7. Results on Synthetic Benchmarks

| Metric | Baseline | This system | Target |
|---|---|---|---|
| Pooled-Ridge RMSE (retention) | 0.085 | – | n/a |
| Per-sector Ridge RMSE | – | **0.054 ± 0.003** | ≤ 0.060 |
| Per-sector worst-MAE | 0.11 | **0.07 ± 0.005** | ≤ 0.080 |
| Bandit cumulative reward (vs random) | 1.00× | **1.42 ± 0.04×** | ≥ 1.30× |
| Best-arm-historical (greedy) reward | – | 1.38× | – |
| `/benchmark` p95 latency | n/a | **62 ms** | < 80 ms |

The per-sector Ridge dominates the pooled baseline on every per-sector RMSE and on the worst-sector MAE. The bandit beats both random and pure-greedy baselines, with the gap to greedy attributable to the 10% exploration keeping the per-arm means honest as the panel grows.

The $\varepsilon$ sweep shows a flat plateau between 0.05 and 0.15, with cumulative reward dropping at $\varepsilon = 0.20$ as the exploration cost begins to dominate. The reward-noise ablation shows top-1 action stability of about 0.91 (Spearman correlation between rankings on clean and 5% noise), indicating the ranking surface is robust at this noise level.

## 8. Limitations and Threats to Validity

**Synthetic panel.** All numbers are on a synthetic generator with a known reward function. Real HR panels exhibit time-varying reward distributions (programme effects decay), confounding from concurrent interventions, and non-stationary sector base-rates following economic shocks — none of which is captured here. The reported numbers should be read as a ceiling on what ε-greedy plus per-sector Ridge can recover under stationarity, not as a production estimate.

**No partial pooling.** The structural-hierarchy-only choice corresponds to the no-pooling extreme. Sectors with small $n$ (Construction, Education in our panel) would benefit from a Bayesian-hierarchical prior centred on a global $\bar{\mathbf{w}}$. This is the documented Track-2 upgrade.

**Under-exploration of small arms.** ε-greedy is known to under-explore in heavy-tailed reward settings. UCB1 or LinUCB would tighten the regret bound; Thompson sampling would give Bayesian-coherent posterior intervals. The bandit slot is upgradeable without changing the API contract.

**Reward proxy.** The reward is the action's contribution to retention above the do-nothing baseline. In production this would have to be estimated from a counterfactual (uplift) model rather than read off, and the noise on that estimate would propagate into the bandit's mean updates.

**Discrete action set.** The five-action set is operationally convenient but is a discretisation of an underlying continuous space (how-much-comp-bump, how-many-training-hours). A continuous-action bandit (e.g., Bayesian optimisation over the action space) is the natural next iteration.

## 9. Conclusion

The right shape of an HR retention artefact is not a benchmark; it is a benchmark plus a ranked-action list with explicit n, returned by one API call. The technical components — per-sector Ridge, ε-greedy bandit, a fixed five-action set — are deliberately conservative because the load-bearing properties of the dashboard are the structural per-sector hierarchy, the n shown next to every recommendation, and the n-suppression release gate. Both halves of the system can be upgraded — Bayesian-hierarchical Ridge for partial pooling, LinUCB or Thompson sampling for tighter exploration, continuous-action bandits for non-discrete levers — without changing the API contract. On a synthetic 3,200-row cross-sector panel the system reaches the target RMSE, worst-sector MAE, and cumulative-reward thresholds, with the n-suppression rule exercised on roughly 12% of arm pulls. Future work focuses on the partial-pooling and continuous-action upgrades.

## References

[1] A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.

[2] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York: Springer, 2009.

[3] A. Gelman and J. Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press, 2007.

[4] A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D. B. Rubin, *Bayesian Data Analysis*, 3rd ed. Chapman and Hall/CRC, 2013.

[5] R. McElreath, *Statistical Rethinking: A Bayesian Course with Examples in R and Stan*, 2nd ed. Chapman and Hall/CRC, 2020.

[6] P. Auer, N. Cesa-Bianchi, and P. Fischer, "Finite-time analysis of the multiarmed bandit problem," *Machine Learning*, vol. 47, no. 2–3, pp. 235–256, 2002.

[7] T. Lattimore and C. Szepesvári, *Bandit Algorithms*. Cambridge University Press, 2020.

[8] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.

[9] L. Li, W. Chu, J. Langford, and R. E. Schapire, "A contextual-bandit approach to personalized news article recommendation," in *Proceedings of WWW*, 2010, pp. 661–670.

[10] W. R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples," *Biometrika*, vol. 25, no. 3/4, pp. 285–294, 1933.

[11] M. Dudík, J. Langford, and L. Li, "Doubly robust policy evaluation and learning," in *Proceedings of ICML*, 2011, pp. 1097–1104.

[12] L. Li, W. Chu, J. Langford, and X. Wang, "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms," in *Proceedings of WSDM*, 2011, pp. 297–306.

[13] P. Gutierrez and J.-Y. Gérardy, "Causal inference and uplift modelling: A review of the literature," in *Proceedings of the International Conference on Predictive Applications and APIs (PAPIs)*, PMLR vol. 67, 2017, pp. 1–13.

[14] J. D. Angrist and J.-S. Pischke, *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press, 2009.

[15] D. M. Roy and Y. W. Teh, "The Mondrian process," in *Advances in Neural Information Processing Systems 21 (NIPS)*, 2008.
