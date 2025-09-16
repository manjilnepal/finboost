# FinSurvival Challenge Starting Kit

Welcome to the **FinSurvival Challenge** starting kit! This competition focuses on advancing deep survival modeling for financial transactions using DeFi (Decentralized Finance) data.

## Competition Overview

The FinSurvival Challenge is a survival analysis competition that predicts time-to-event outcomes in financial transactions. You'll work with real DeFi data from the Aave protocol to predict when users will experience events like loan defaults, repayments, or liquidations.

## Key Competition Details

- **16 Prediction Tasks**: Each participant must create models for 16 unique event pairs
- **Event Types**: Borrow, Deposit, Repay, Withdraw, Liquidated
- **Evaluation Metric**: Concordance Index (C-index) for survival analysis
- **Dataset**: 21.8M+ records with 90 features from Aave V3 protocol (March 12, 2022 - July 2, 2024)
- **Submission Format**: 16 pre-trained models + 1 preprocessing script

## Getting Started

### 1. **Review the Demo Notebook**
**Start with the `prediction_model_demo.ipynb` notebook** - it provides a complete workflow showing how to:
- Load and preprocess the survival datasets
- Train Cox Proportional Hazards models (as a baseline example)
- Evaluate models using C-index
- Package submissions correctly

**Note:** The Cox Proportional Hazards model is provided as a **baseline example only**. Participants are encouraged to experiment with various other survival models such as:
- Random Survival Forests
- Deep Learning Survival Models (DeepSurv, DeepHit)
- Accelerated Failure Time (AFT) models
- Gradient Boosting Survival Models
- Neural Networks for Survival Analysis

### 2. **Key Files in This Kit**

- **`prediction_model_demo.ipynb`**: Complete example workflow (START HERE!)
- **`preprocessing.py`**: Customizable data preprocessing pipeline
- **`survival_metrics.py`**: C-index evaluation implementation

### 3. **Competition Structure**

The competition has two phases:
- **Development Phase** (Sep 15 - Oct 14, 2025): 5 submissions/day, 100 total
- **Final Phase** (Oct 15 - Oct 20, 2025): 2 submissions total, evaluated against a holdout test set

### 4. **Event Pairs to Model**

You need to create models for these 16 event transitions:
- Borrow → Deposit, Repay, Withdraw, Liquidated
- Deposit → Borrow, Repay, Withdraw, Liquidated
- Repay → Borrow, Deposit, Withdraw, Liquidated
- Withdraw → Borrow, Deposit, Repay, Liquidated

### 5. **Submission Requirements**

Your submission must include:
- **16 pre-trained model files** (`.pkl` format) named as `<index_event>_<outcome_event>.pkl` (e.g., `Borrow_Repay.pkl`, `Deposit_Liquidated.pkl`)
- **1 preprocessing script** (`preprocessing.py`) with a function named `preprocess`
- **1 model definition file** (`model.py`) with the class definition of your model
- All packaged in a single ZIP file

## Next Steps

1. **Explore the sample models:**
   - **`prediction_model_demo_cox.ipynb`** - Cox Proportional Hazards model example
   - **`prediction_model_demo_aft.ipynb`** - Accelerated Failure Time (AFT) model example
2. **Run through the complete examples** to understand the workflow
3. **Customize `preprocessing.py`** to improve your feature engineering
3. **Experiment with different survival models** - the Cox model is just a starting point! Try advanced models like Random Survival Forests, DeepSurv, or gradient boosting approaches
4. **Submit your models** through the Codabench platform

## Resources

- **Official Competition Website**: [https://finsurvival.github.io/](https://finsurvival.github.io/) - For detailed information about the challenge, evaluation protocol, and timeline
- **Competition Website**: Check the overview, data, and evaluation pages
- **Survival Analysis**: Familiarize yourself with survival analysis concepts
- **DeFi Concepts**: Understand Aave V3 protocol mechanics for better feature engineering

Good luck with the competition! 🚀