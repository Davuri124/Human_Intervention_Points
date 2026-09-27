# Learning When to Intervene

**A Reinforcement Learning Framework for Adaptive Human Oversight in Automated AI Pipelines**

Autonomous AI pipelines — spanning perception, decision, and action stages — increasingly operate with only intermittent human oversight. Deciding *when* a human should be pulled into the loop (an **intervention point**) is a sequential decision problem: intervening too often erodes the efficiency gains of automation, while intervening too rarely lets costly errors propagate uncaught.

This project formulates intervention-point detection as a **Markov Decision Process** and trains a **Proximal Policy Optimization (PPO)** agent that observes real, model-derived uncertainty and risk signals — rather than hand-tuned heuristics — to decide when to escalate control to a human.

## Why this matters

Existing approaches are typically static: a fixed confidence threshold below which a case is escalated. These threshold rules ignore the sequential structure of a pipeline (uncertainty trends, accumulated error, remaining intervention budget) and need manual re-tuning whenever the task distribution shifts. This framework instead learns the intervention policy end-to-end.

## Project phases

The system was developed and validated across an eight-phase experimental program:

Phase A: MDP formulation driven by real uncertainty (predictive entropy, MC dropout variance) instead of synthetic proxies
Phase B: Curriculum learning for faster, more stable convergence
Phase C: Multi-pipeline generalization, cost-aware intervention, and a theoretical regret bound
Phase D: Model-Agnostic Meta-Learning (MAML) for few-shot adaptation to unseen pipeline types
Phase E: Robustness under simulated human-operator profiles and adversarial perturbations
Phase F / F2: Zero-shot validation on four external real-world tabular datasets (no retraining)
Phase G: Ablations against rule-based thresholds and contextual-bandit baselines
Phase H: Full statistical validation battery (bootstrap CIs, effect sizes, power analysis) plus SHAP explainability

## Key results

- Across 200+ experimental configurations, the learned policy outperforms random, always-intervene, never-intervene, threshold-based, and contextual-bandit baselines by wide, statistically validated margins (Cohen's d up to ~19, p < 0.001).
- The MAML-initialized policy adapts to a previously unseen pipeline within five gradient updates.
- Validated with zero retraining on four external real-world datasets (Credit Card Fraud, Heart Disease, Diabetes, Bank Marketing), consistently beating a random-intervention baseline.
- The policy retains 92% or more of clean performance under the largest tested adversarial perturbation.
- SHAP analysis confirms the policy relies on interpretable, domain-appropriate signals — uncertainty and error risk dominate attribution mass.

## Repository structure

agents/ - RL agents, training scripts, MAML/curriculum/threshold baselines
environments/ - Pipeline/MDP environment definitions
pipelines/ - Pipeline archetypes (medical, fraud, NLP, navigation, etc.)
results/ - Experiment outputs, figures, evaluation logs
data/ - Datasets used for training/evaluation (not tracked, see below)
main.py - Entry point
test_setup.py - Environment/setup sanity checks
README.md - This file

Note: The data folder is excluded from version control due to file size limits. See the Data section below for how to obtain it.

## Setup

git clone https://github.com/your-username/intervention_point_detection.git
cd intervention_point_detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Usage

python main.py

Run the setup sanity checks with:

python test_setup.py

## Data

This project uses several public tabular datasets for real-world validation (Credit Card Fraud, Heart Disease, Diabetes, Bank Marketing). Due to size, raw data files are not included in this repository. Place your dataset files under data/ before running training or evaluation scripts.

## Method summary

Agent: PPO (two-layer, 64-unit tanh network), gamma 0.99, learning rate 3e-4, batch size 1000
State: 8 features - uncertainty, error risk, complexity, step progress, remaining intervention budget, uncertainty trend, error accumulation, pipeline phase
Reward: step penalty plus large penalty for missed errors plus moderate penalty for wasted interventions plus bonus for correct interventions
Evaluation: two-sample t-test / Mann-Whitney U test, bootstrap 95% confidence intervals (5,000 resamples), Cohen's d effect sizes, statistical power analysis

## License

Add your preferred license here (e.g. MIT).

## Citation

If you use this work, please cite the accompanying paper (details to be added upon publication).
