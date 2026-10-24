# Credit Card Fraud Detection Using Machine Learning

A comparative machine learning project using the BankSim dataset to detect fraudulent financial transactions, evaluate model performance under class imbalance, and estimate potential financial savings.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem in which fraudulent transactions represent only a small proportion of all transactions. A useful detection system must identify suspicious activity while limiting false alarms and the operational cost of reviewing legitimate transactions.

This project investigates whether machine learning models can improve fraud detection over simple, rule-based baselines. It compares predictive performance, evaluates business impact, and uses a time-based validation strategy to reduce information leakage.

The analysis is conducted using the **BankSim synthetic banking transaction dataset**.

## Objectives

- Explore transaction patterns and identify characteristics associated with fraudulent activity.
- Investigate class imbalance and differences across transaction categories, merchants, and customer demographics.
- Engineer behavioral features using historical transaction information.
- Compare Logistic Regression, RBF Support Vector Machine (SVM), and Decision Tree models.
- Benchmark machine learning models against amount-based and category-based baselines.
- Select classification thresholds using validation data.
- Estimate net financial savings after accounting for transaction losses and review costs.
- Assess uncertainty, feature dependence, and the limitations of the resulting models.

## Dataset

The notebook uses the BankSim dataset, containing **594,643 transactions** across 180 time steps, representing approximately six months of simulated activity.

| Characteristic | Description |
|---|---|
| Dataset | BankSim |
| Total transactions | 594,643 |
| Fraudulent transactions | 7,200 |
| Fraud prevalence | Approximately 1.21% |
| Target variable | `fraud` |
| Primary transaction features | Amount, category, merchant, customer attributes |
| Temporal information | `step`, used to preserve chronological ordering |

The dataset is synthetic. Its fraud patterns and predictive relationships may differ significantly from those found in real banking systems.

The notebook expects the dataset file `bs140513_032310.csv` to be available in the working directory.

## Technologies and Libraries

- **Python** — primary programming language
- **Pandas and NumPy** — data manipulation and numerical computation
- **Matplotlib and Seaborn** — exploratory data analysis and visualization
- **SciPy** — statistical testing
- **Scikit-learn** — preprocessing, model training, hyperparameter evaluation, and performance metrics
- **Jupyter Notebook** — interactive analysis and experimentation

## Machine Learning Models

The project compares three supervised learning algorithms against two simple baselines.

| Model | Purpose |
|---|---|
| Logistic Regression | Establish a linear classification benchmark |
| Support Vector Machine (RBF kernel) | Capture nonlinear relationships |
| Decision Tree | Model nonlinear patterns and feature interactions |
| Price baseline | Rank transactions using transaction amount |
| Category baseline | Rank transactions using fraud rates estimated from training data |

The SVM is trained on a subset of the training data because of its computational cost. The subset includes fraudulent transactions from the training set and 40,000 randomly sampled legitimate transactions under the standard configuration.

## Methodology

### 1. Data Cleaning and Preparation

The notebook performs data quality checks and prepares the dataset for analysis.

Key steps include:

- Removing unnecessary quotation marks from categorical values.
- Checking chronological ordering and identifying constant location-related columns.
- Removing columns that provide no useful variation.
- Standardizing category labels.
- Handling unknown age values.
- Preserving transaction order for subsequent time-based splitting.

### 2. Time-Based Train, Validation, and Test Split

Instead of randomly splitting transactions, the notebook preserves their chronological order.

- **Training set:** used for exploratory analysis, feature preparation, and model fitting.
- **Validation set:** used for model comparison, hyperparameter selection, and decision-threshold optimization.
- **Test set:** reserved for final performance evaluation after the model configuration and thresholds have been frozen.

This approach better reflects the intended use case of predicting future transactions from historical data.

### 3. Exploratory Data Analysis

Exploratory analysis focuses on the training data and investigates:

- Fraud prevalence and class imbalance.
- Fraud rates across transaction categories.
- Transaction amount distributions.
- Customer age and gender patterns.
- Merchant-level fraud behavior.
- Repeated transactions and customers affected by fraud.
- Changes in transaction distributions between training and validation periods.

The findings inform feature engineering and model design.

### 4. Feature Engineering

The notebook constructs transaction-level features designed to capture both transaction characteristics and customer behavior.

Examples include:

| Feature | Description |
|---|---|
| `log_amount` | Log-transformed transaction amount |
| `amt_z_in_category` | Relative amount anomaly within a transaction category |
| `log_n_prior_txn` | Log-transformed number of previous customer transactions |
| `is_first_txn` | Indicates whether the transaction is the customer's first observed transaction |
| `log_amt_to_cust_mean` | Transaction amount relative to the customer's historical average |
| `log_amt_to_cust_max` | Transaction amount relative to the customer's historical maximum |
| `is_new_merchant` | Indicates a first observed transaction by the customer at a merchant |
| `is_new_category` | Indicates a first observed transaction by the customer in a category |

Historical behavioral features are calculated using prior transactions rather than future observations. Customer identifiers, the time-step column, and direct indicators of previously observed fraud are excluded from the model features.

### 5. Preprocessing and Model Training

The modeling pipeline uses:

- `StandardScaler` for numerical features.
- `OneHotEncoder` for categorical variables.
- Training-only fitting of preprocessing transformations.
- A small hyperparameter search using the validation set.
- Experiments comparing static features with additional behavioral features.
- Evaluation of class weighting as a way to address class imbalance.

The SVM uses a reduced training sample to manage computational requirements.

### 6. Evaluation Metrics

Because fraudulent transactions account for approximately 1.21% of the dataset, accuracy alone would be misleading.

The notebook emphasizes the following metrics:

- **PR-AUC (Average Precision):** measures ranking performance with a focus on the minority fraud class.
- **Precision:** measures the proportion of flagged transactions that are fraudulent.
- **Recall:** measures the proportion of fraudulent transactions detected.
- **F1-score:** balances precision and recall at a selected threshold.
- **ROC-AUC:** provides a supplementary measure of ranking performance.
- **Number of alerts:** estimates the operational workload created by the detection system.
- **Net financial savings:** estimates the financial benefit after accounting for prevented fraud losses and review costs.

F1-optimized thresholds and cost-optimized thresholds are selected using validation data. The chosen thresholds are then frozen before the final test evaluation.

### 7. Financial Impact Analysis

The notebook estimates the potential financial benefit of flagging fraudulent transactions.

The calculation accounts for the value of detected fraud, the assumed effectiveness of preventing those losses, and the cost of reviewing alerts.

**Estimated net savings:**

Net savings = Prevented fraud losses − Alert review costs

The standard analysis assumes a review cost of $5 per alert and a specified fraud-prevention effectiveness. Sensitivity analysis varies both assumptions to examine how business conclusions change under different operating conditions.

These estimates are scenario-based, not measured savings from a real financial institution.

### 8. Model Interpretation and Robustness

Additional analyses examine:

- False positives and missed fraudulent transactions.
- The contribution of different feature groups.
- Model dependence on merchant identifiers and demographic features.
- Performance within higher-risk transaction categories.
- Bootstrap confidence intervals using contiguous blocks of test transactions.
- Sensitivity of estimated financial savings to review costs and fraud-prevention effectiveness.

The test set is used for final reporting rather than iterative model selection.

## Results

The notebook produces a comparative evaluation of all three machine learning models and both baselines.

The principal results include:

- PR-AUC and F1-score on the held-out test set.
- Precision, recall, and the number of alerts at selected thresholds.
- Estimated net financial savings and the percentage of baseline fraud losses avoided.
- Bootstrap confidence intervals for key metrics.
- Sensitivity analysis under alternative cost and prevention assumptions.
- Performance on higher-risk transaction categories.

**Final model performance should be reported using the actual outputs generated by running the notebook.** No numerical test results or winning model are asserted here because they should be taken directly from the notebook's executed results.

## How to Run the Project

### Prerequisites

Install Python and Jupyter Notebook or JupyterLab.

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### Setup

1. Download or clone the project repository.
2. Place `bs140513_032310.csv` in the working directory alongside the notebook.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `Fraud_Detection(4).ipynb`.
5. Run the notebook cells sequentially from beginning to end.

The notebook uses a fixed random seed for reproducibility. Runtime depends on the machine, dataset, and selected model configurations.

## Limitations

Several limitations should be considered when interpreting the findings:

1. **Synthetic data:** BankSim may contain simpler fraud patterns than real-world transaction data.
2. **Class imbalance:** The observed fraud prevalence may not represent the prevalence encountered in production.
3. **SVM sampling:** The SVM uses fewer legitimate transactions during training, so its training setup differs from those of the other models.
4. **Assumed financial costs:** Review cost and prevention effectiveness are modeled assumptions that require real operational data for practical deployment.
5. **Limited test period:** A single held-out time window cannot establish long-term stability.
6. **Merchant dependence:** Merchant identifiers may be predictive in the simulated dataset but may generalize poorly when fraud patterns or merchants change.
7. **Temporal features:** Removing the time-step column limits the use of explicit time-based behavioral signals.
8. **Deployment requirements:** A production system would require monitoring, threshold recalibration, data-quality controls, and periodic retraining.

The results should therefore be interpreted as an experimental evaluation, not as evidence of production-ready fraud prevention.

## Future Improvements

Potential next steps include:

- Evaluating gradient-boosting models such as XGBoost or LightGBM.
- Testing on additional time periods and datasets with realistic fraud prevalence.
- Developing richer rolling-window behavioral features.
- Calibrating model scores into reliable fraud probabilities.
- Optimizing thresholds according to review-team capacity and real operational costs.
- Monitoring data drift, fraud prevalence, alert volume, and model performance.
- Evaluating fairness and the necessity of demographic features.
- Reassessing performance when merchants, fraud strategies, and transaction behavior change.

## Conclusion

This project provides an end-to-end workflow for fraud detection, from exploratory data analysis and leakage-aware feature engineering to model comparison, threshold selection, and business-oriented evaluation.

By comparing machine learning models with simple baselines and evaluating both predictive quality and estimated financial impact, the notebook demonstrates a more practical approach to fraud detection than relying on accuracy alone.

The central objective is to identify a model that detects fraudulent transactions effectively while maintaining a manageable alert volume and delivering positive estimated net savings under clearly stated assumptions.
