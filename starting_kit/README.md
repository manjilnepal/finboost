# **FinSurvival Challenge Starting Kit**

Welcome to the **FinSurvival Challenge** starting kit\! This competition focuses on advancing deep survival modeling for financial transactions using DeFi (Decentralized Finance) data.

## **Competition Overview**

The FinSurvival Challenge is a survival analysis competition that predicts time-to-event outcomes in financial transactions. You'll work with real DeFi data from the Aave protocol to predict when users will experience events like loan defaults, repayments, or liquidations.

## **Key Competition Details**

* **16 Prediction Tasks**: Each participant must create models for 16 unique event pairs.  
* **Event Types**: Borrow, Deposit, Repay, Withdraw, Liquidated.  
* **Evaluation Metric**: Concordance Index (C-index) for survival analysis.  
* **Dataset**: 21.8M+ records with 90 features from the Aave V3 protocol (March 12, 2022 \- July 2, 2024).  
* **Submission Format**: 16 prediction files (.csv format), one for each task.

## **Getting Started**

### **1\. Review the Starter Notebooks**

**Start with the provided starter notebooks (Starter\_Notebook\_AFT.ipynb and Starter\_Notebook\_Cox.ipynb)**. They provide a complete workflow showing how to:

* Load the training and test feature datasets.  
* Train a survival model (AFT and Cox models are provided as baseline examples).  
* Generate predictions on the test set.  
* Save the predictions in the correct CSV format.  
* Package all 16 prediction files into a submission.zip file.

**Note:** The example models are provided as **baselines only**. You are encouraged to experiment with any survival model you choose to generate your predictions.

### **2\. Key Files in This Kit**

* **Starter\_Notebook\_AFT.ipynb**: Complete example workflow for the Weibull AFT model.  
* **Starter\_Notebook\_Cox.ipynb**: Complete example workflow for the CoxPH model.  

### **3\. Competition Structure**

The competition has two phases:

* **Development Phase** (Sep 15 \- Oct 14, 2025): 5 submissions/day, 100 total. You will be evaluated on a hidden validation set.  
* **Final Phase** (Oct 15 \- Oct 20, 2025): 2 submissions total, evaluated against a final holdout test set.

### **4\. Event Pairs to Predict**

You need to generate prediction files for these 16 event transitions:

* Borrow → Deposit, Repay, Withdraw, Liquidated  
* Deposit → Borrow, Repay, Withdraw, Liquidated  
* Repay → Borrow, Deposit, Withdraw, Liquidated  
* Withdraw → Borrow, Deposit, Repay, Liquidated

### **5\. Submission Requirements**

Your submission must be a single **submission.zip** file containing:

* **16 prediction files** in CSV format.  
* Each file must be named as \<index\_event\>\_\<outcome\_event\>.csv (e.g., Borrow\_Repay.csv, Deposit\_Liquidated.csv).  
* Each CSV file should contain a single column of numerical predictions, with no header.

## **Next Steps**

1. **Choose a starter notebook** to begin (AFT or Cox).  
2. **Run through the complete example** to understand the workflow.  
3. **Customize the preprocess function** inside the notebook to improve your feature engineering.  
4. **Experiment with different survival models** to generate better predictions. The examples are just a starting point\! Try advanced models like Random Survival Forests, DeepSurv, or gradient boosting approaches.  
5. **Generate your prediction files** and upload your submission.zip to the CodaBench platform.

## **Resources**

* **Official Competition Website**: [https://finsurvival.github.io/](https://finsurvival.github.io/) \- For detailed information about the challenge, evaluation protocol, and timeline.  
* **Competition Platform**: Check the overview, data, and evaluation pages on CodaBench.  
* **Survival Analysis**: Familiarize yourself with survival analysis concepts if you are new to the field.  
* **DeFi Concepts**: Understand Aave V3 protocol mechanics for better feature engineering.

Good luck with the competition\! 🚀
