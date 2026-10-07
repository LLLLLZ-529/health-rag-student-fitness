# CHIL 2027 Paper Outline — Detailed Blueprint

**Main line (one sentence):** Lacking any systematic benchmark of modern tabular models in student physical-health testing, we benchmark 16 contemporary tabular models on a 9-level health-state labeling scheme, show that trajectory features are the decisive lever while explicit temporal models add nothing on ultra-short sequences, establish the feasibility boundary of cross-grade early degradation warning, and close the loop with an integrated classification → early-warning → RAG-based contextual-bandit recommendation pipeline.

**Title candidates**
1. *Beyond Decision Trees: A 16-Model Benchmark of Modern Tabular Models and an Integrated Classification–Early-Warning–Recommendation Pipeline for Student Physical Health*
2. *Trend Is All You Need? Benchmarking Tabular Models for Student Physical-Fitness Classification, Early Degradation Prediction, and RAG-Based Personalized Recommendation*
3. *From Benchmark to Intervention: Modern Tabular Models, Short-Sequence Early Warning, and RAG–Bandit Recommendation on a Multi-Grade Student Health Dataset*

**Paper promise.** We use a multi-grade student physical-fitness cohort (HI9 nine-class labels, 57 features) and 16 contemporary tabular models to establish evidence-level findings on model ranking, the role of trajectory features, the feasibility boundary of early warning, and an end-to-end recommendation pipeline — a *benchmark + application* contribution, not a methodological one.

**Not claimed.** We do not propose a new tabular model, claim state-of-the-art, establish causal effects of the recommended interventions, or validate clinical/field deployment; labels are rule-derived from the Health Index (HI), results are observational and single-cohort, and no prospective validation has been performed.

**Contributions (3–4).**
- **C1 (Benchmark).** First systematic comparison of 16 modern tabular models — including foundation-model/tab-pretrained approaches (TabPFN-2.5, TabICLv2, TabM, ModernNCA, SupCon) and classical baselines (RandomForest, RBF-SVM, logistic, BPNN) — on a real student physical-health classification task (9 HI9 classes), reporting accuracy, Macro-F1, clustering metrics (JC/FMI/RI/DBI/DI), and efficiency (params/latency).
- **C2 (Key feature finding).** A controlled feature ablation showing trend/slope features dominate performance (7-D raw ≈ 49% → 14-D raw+slope ≈ 82% → 57-D HI pipeline ≈ 85%), and a negative result that explicit temporal models (GRU/LSTM/CNN) and handcrafted delta/slope/accel features yield no gain on 4–8-point sequences.
- **C3 (Early-warning feasibility boundary).** Cross-grade binary decline classification is feasible (≈82–84% acc), whereas nine-class fine-grained prediction remains hard (≈53% cross-grade); performance rises monotonically with observation horizon (g1→g4 ≈ 44% → g12→g4 ≈ 46% → g123→g4 ≈ 54%).
- **C4 (Integrated pipeline).** An end-to-end system combining classification, early degradation warning, and TF-IDF retrieval + contextual-bandit recommendation over a 2,674-chunk health-knowledge base (4 intervention strategies; 8/9 student groups converge), demonstrating how a benchmark finding translates into a deployable health-monitoring application.

---

## 1. Introduction
*Argument task:* Establish why this problem matters and what is missing, then stake the claim ceiling so the reader knows exactly what the paper will and will not deliver. *Materials:* Motivating statistics on the scale of student physical-health monitoring; the literature gap (all prior fitness-ML work uses only decision trees/RF/BPNN; tabular foundation models have not reached this vertical); the three experimental results summarized as evidence. *Transition:* Ends with the bulleted contributions and a one-paragraph roadmap, so Section 2 can credibly position the work against prior art.
- 1.1 Motivation: student physical testing as a routine, large-scale, *tabular* health-prediction problem.
- 1.2 Gap: no modern tabular-model benchmark; undiscussed role of trajectory features; unexplored fusion of RAG and bandits in this setting.
- 1.3 Contributions (C1–C4 above) and paper roadmap.

## 2. Related Work
*Argument task:* Show that the benchmark blank, the short-sequence methodological gap, and the RAG/bandit fusion blank are real, and locate our positioning in the live debates rather than merely listing papers. *Materials:* The 35 verified references from the literature survey, organized into five themes mirroring the survey's core judgments. *Transition:* The section closes by explicitly deriving the three blanks our experiments target, which Section 3 then operationalizes into a concrete task and labels.
- 2.1 Deep learning for tabular data and the GBDT-vs-tabular-DL debate — Grinsztajn et al. (2022), TabArena (Erickson et al., 2025), TabPFN as tabular foundation model (Hollmann et al., Nature 2025), and the remaining verified references on tabular ICL/pretraining.
- 2.2 Machine learning in physical fitness / sports-science testing — the decision-tree/RF/BPNN lineage that motivates C1.
- 2.3 Early deterioration / early-warning prediction in health — Hyland et al. (2020) vs. Shah et al. (2022) prospective-validation caution; ICU high-frequency sampling vs. our 4–8-point setting.
- 2.4 Retrieval-augmented and bandit-based health recommendation — Hofer (2026) and Newby et al. (2025, JAMA RCT) on bandit-vs-LLM intervention recommendation; verified RAG-in-health references.
- 2.5 Trajectory / slope features in longitudinal health classification — the undiscussed role our ablation (C2) addresses.
- **Table:** related-work positioning matrix (*Study* | *Domain* | *Model family* | *Uses trend features?* | *Short-sequence?* | *Recommendation?*), ending on the row that is ours alone.

## 3. Problem Formulation and Dataset
*Argument task:* Pin down the object of study and the labeling scheme before any model is compared, so downstream rankings are interpretable. *Materials:* the HI9 label construction (Level: total_score <60 / 60–80 / ≥80, crossed with Trend: HI_slope <−0.1 / ±0.1 / >0.1 → 9 classes); the cohort, 57 features, and 4–8 observation time points per student. *Transition:* Because the label definition is the load-bearing assumption for every later number, this section must precede the pipeline and all three experiments.
- 3.1 Study setting and cohort.
- 3.2 HI9 label definition (Level × Trend) and the nine-class target.
- 3.3 Feature set (raw measurements, derived slope/trajectory features, 57-D) and per-grade statistics.
- **Table 1:** cohort and feature statistics; **Table 2:** Level × Trend contingency of the 9 HI9 classes.

## 4. Method: Integrated Pipeline Overview
*Argument task:* Present the three-module system once, at the architecture level, so that Sections 5–7 read as deep-dives into modules rather than as disconnected experiments. *Materials:* a single schematic connecting (i) nine-class health-state classification, (ii) cross-grade early degradation warning, and (iii) RAG retrieval + contextual-bandit intervention selection. *Transition:* Deliberately result-free — it sets up what each module is for; the evidence for each is reported in order, and the benchmark result (M1) is the natural first dependency because M2 reuses its model pool.
- 4.1 Three-module architecture and data flow.
- 4.2 Module interfaces: classification output → warning input → bandit context.
- **Fig. 1:** end-to-end pipeline overview (classification → early warning → RAG+Bandit recommendation).

## 5. Experiment 1 — Tabular Model Benchmark (M1)
*Argument task:* Answer "which modern tabular model fits this task, and what actually drives performance?" *Materials:* 16 models on 57 features and 9 HI9 classes; accuracy/Macro-F1 plus clustering metrics (JC/FMI/RI/DBI/DI) and an efficiency profile (parameter count, inference latency). *Transition:* Establishes both the model pool reused in M2 and the trend-feature finding (C2), which the Discussion then elevates.
- 5.1 Task, evaluation protocol, and metrics (Acc, Macro-F1, clustering metrics).
- 5.2 Baselines and model pool (TabM, TabPFN-2.5, ModernNCA, TabICLv2, SupCon, RandomForest, RBF-SVM, logistic, BPNN, and the remaining 16-model roster).
- 5.3 Main results and ranking — TabM (0.859) ≈ TabPFN-2.5 (0.856) ≈ ModernNCA (0.855) ≈ TabICLv2/SupCon (0.845) well above RandomForest (0.630).
- 5.4 Trend-feature ablation: 7-D raw (≈49%) → 14-D raw+slope (≈82%) → 57-D HI (≈85%).
- 5.5 Efficiency vs. accuracy trade-off — ModernNCA lightest (0.116M params, 0.173 ms) vs. ~100M pretrained TabPFN-family.
- **Table 3:** full 16-model results; **Table 4:** efficiency; **Fig. 2:** ranking bar chart; **Fig. 3:** feature-ablation bars; **Fig. 4:** params-vs-latency scatter.

## 6. Experiment 2 — Early Degradation Prediction (M2)
*Argument task:* Answer "how early, and how granularly, can degradation be warned from few time points?" *Materials:* g1+g2→g4 nine-class prediction; the binary "declining" task; a no-leakage cross-grade split; observation-horizon ablation; and a temporal-model comparison (GRU/MLP/CNN/LSTM) plus the MTG-Net delta/slope/accel kill test. *Transition:* Consumes the M1 model pool and produces the feasibility boundary that motivates the recommendation module.
- 6.1 Task definition: predict a later grade's state from earlier grades.
- 6.2 Nine-class vs. binary-decline prediction (binary ≈0.82 acc; TabICLv2/TabPFN-2.5 tied top).
- 6.3 Cross-grade generalization without leakage (9-class ≈53%; binary ≈84%).
- 6.4 Prediction-horizon ablation: g1→g4 ≈44% → g12→g4 ≈46% → g123→g4 ≈54%.
- 6.5 Why explicit temporal modeling fails: GRU/LSTM/CNN vs. tabular baselines on 4 points; MTG-Net kill test shows handcrafted delta/slope/accel adds no gain.
- **Table 5:** M2 results (9-class vs. binary, in-sample vs. cross-grade); **Fig. 5:** horizon monotonic curve; **Fig. 6:** temporal-vs-tabular comparison.

## 7. Experiment 3 — RAG-Based Health Recommendation (M3)
*Argument task:* Show how the classification/warning outputs become an actionable, learnable intervention recommendation. *Materials:* a 2,674-chunk knowledge base (exercise prescription, diet/nutrition, fitness standards, health policy); TF-IDF retrieval; a contextual bandit over 4 strategies (exercise / diet / combined / maintain); convergence across 9 student groups. *Transition:* This is the application endpoint of the pipeline; its results feed directly into the deployment considerations in Discussion.
- 7.1 Knowledge-base construction and chunking.
- 7.2 Retrieval (TF-IDF) and contextual-bandit strategy selection.
- 7.3 Convergence and per-group results — 8/9 groups converge; the small high-level-declining stratum (93 students) does not.
- **Table 6:** convergence summary by group; **Fig. 7:** bandit convergence curves by strategy.

## 8. Discussion
*Argument task:* Synthesize the three experiments into a few defensible claims and connect them to the open debates, without overreaching. *Materials:* the benchmark ranking, the trend-feature ablation, the short-sequence negative result, and the pipeline outcome. *Transition:* Moves from "what we measured" to "what it means and where the limits are," handing off to the dedicated limitations/ethics section.
- 8.1 Trajectory features as the central lever for physical-health classification.
- 8.2 What the benchmark implies for the GBDT-vs-tabular-DL debate in this vertical.
- 8.3 Why ultra-short sequences defeat explicit temporal modeling (and what that implies for method choice).
- 8.4 Deployment considerations: edge efficiency (ModernNCA) vs. pretrained accuracy; the non-convergent stratum as a known failure mode.

## 9. Limitations and Ethics
*Argument task:* State the boundary conditions and the health-AI equity/privacy obligations CHIL expects, explicitly and without softening the load-bearing assumptions. *Materials:* the HI rule-derived labels, single-cohort design, and the observational nature of all three experiments. *Transition:* Concludes the evidentiary discussion so the final section can state the contribution crisply.
- 9.1 Methodological limits: rule-derived (non-clinical) labels, single cohort, no prospective validation, confounders across school/gender/region.
- 9.2 Fairness and distribution shift: performance may differ across demographic/school strata; the small non-convergent group signals sample-size inequity in recommendation.
- 9.3 Privacy and ethics: minor students' sensitive health data, retrieval provenance, and the absence of causal evidence that recommended interventions improve outcomes.

## 10. Conclusion
*Argument task:* Restate the benchmark-plus-application contribution within the stated claim ceiling. *Materials:* C1–C4 recap. No new results.

## References / Appendix
References in author–year form with verified links; Appendix for full hyperparameter grids, per-class confusion matrices, and the complete 16-model roster detail not shown in Table 3.

---

### Figure and Table Plan (consolidated)

| # | Type | Section | Content |
|---|---|---|---|
| Fig. 1 | Diagram | 4 | End-to-end pipeline: classification → early warning → RAG+Bandit recommendation |
| Fig. 2 | Bar chart | 5 | 16-model ranking by accuracy/Macro-F1 |
| Fig. 3 | Bar chart | 5 | Trend-feature ablation: 7-D / 14-D / 57-D |
| Fig. 4 | Scatter | 5 | Accuracy vs. params/latency (efficiency trade-off) |
| Fig. 5 | Line chart | 6 | Prediction-horizon monotonic trend (g1→g4, g12→g4, g123→g4) |
| Fig. 6 | Bar chart | 6 | Temporal models (GRU/MLP/CNN/LSTM) vs. tabular baselines |
| Fig. 7 | Curve | 7 | Bandit convergence over 4 strategies across 9 groups |
| Table 1 | Table | 3 | Cohort and feature statistics |
| Table 2 | Table | 3 | HI9 Level × Trend contingency (9 classes) |
| Table 3 | Table | 5 | Full 16-model M1 results (Acc, Macro-F1, JC/FMI/RI/DBI/DI) |
| Table 4 | Table | 5 | Efficiency: params and latency per model |
| Table 5 | Table | 6 | M2: 9-class vs. binary, in-sample vs. cross-grade |
| Table 6 | Table | 7 | Bandit convergence summary by student group |
| Table 7 | Table | 2 | Related-work positioning matrix |

### Dependency check
Every section earns its place: the HI9 labeling (3) is the load-bearing assumption for all downstream numbers; the pipeline (4) frames the three modules without preemption; M1 (5) both answers the benchmark question and supplies the model pool reused in M2 (6); M2's feasibility boundary justifies the warning module that feeds M3 (7); Discussion (8) and Limitations/Ethics (9) must precede Conclusion (10) so the closing claim stays inside the stated ceiling. No new method, causal claim, or SOTA assertion is introduced anywhere outside these sections.
