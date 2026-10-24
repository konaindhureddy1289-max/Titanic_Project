# Titanic Survival Analysis & Prediction
### From Exploratory Data Analysis to a Reusable Machine Learning Pipeline

An end-to-end data science project using the Titanic passenger dataset. The project covers data cleaning, exploratory data analysis (EDA), feature preprocessing, classification model comparison, hyperparameter tuning, and saving a reusable machine learning pipeline.

## Key Results

| Metric | Result |
|---|---:|
| Default Random Forest Accuracy | 81.6% |
| Precision | 0.800 |
| Recall | 0.696 |
| F1 Score | 0.744 |
| ROC-AUC | 0.830 |
| Test Set Size | 179 passengers |

**Key finding:** Survival was strongly associated with sex and passenger class. Approximately 74.0% of female passengers survived, compared with 18.9% of male passengers.

> Built as part of my AI/ML capstone project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Stage 1: Exploratory Data Analysis](#stage-1-exploratory-data-analysis)
- [Stage 2: Machine Learning](#stage-2-machine-learning)
- [Model Evaluation](#model-evaluation)
- [Class Imbalance Handling](#class-imbalance-handling)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Error Analysis](#error-analysis)
- [Regression Side Task](#regression-side-task)
- [Using the Saved Pipeline](#using-the-saved-pipeline)
- [Installation and Usage](#installation-and-usage)
- [Tech Stack](#tech-stack)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Project Overview

This project has two main stages:

1. **Exploratory Data Analysis (EDA):** Clean the dataset, investigate missing values and outliers, visualize distributions, and identify survival patterns.
2. **Machine Learning:** Build and compare classification models, handle class imbalance, tune hyperparameters, evaluate model performance, and save a reusable prediction pipeline.

A secondary task explores multivariate linear regression to predict passenger fares and analyze residuals.

## Problem Definition

The primary objective is to predict whether a Titanic passenger survived.

- `survived = 1`: Passenger survived.
- `survived = 0`: Passenger did not survive.

### Features Used

- Passenger class (`pclass`)
- Sex (`sex`)
- Age (`age`)
- Siblings or spouse aboard (`sibsp`)
- Parents or children aboard (`parch`)
- Fare (`fare`)
- Port of embarkation (`embarked`)

The secondary objective is to predict ticket fare using passenger attributes and evaluate regression performance.

## Dataset

The dataset was loaded using `seaborn.load_dataset("titanic")` and saved locally as `titanic.csv`.

| Property | Details |
|---|---|
| Original dataset size | 891 rows × 15 columns |
| After EDA cleaning | 889 rows × 14 columns |
| Target variable | `survived` |
| Approximate survival rate | 38.4% |
| Dataset file | `titanic.csv` |

## Repository Structure

```text
.
├── titanic.csv
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic_full_pipeline.joblib
├── requirements.txt
└── README.md
```

- `titanic.csv` — Local copy of the dataset.
- `01_eda.ipynb` — Data cleaning and exploratory analysis.
- `02_modeling.ipynb` — Model training, evaluation, tuning, and pipeline saving.
- `titanic_full_pipeline.joblib` — Saved end-to-end model pipeline.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

*Ensure the notebook filenames match the files committed to your repository.*

---

## Stage 1: Exploratory Data Analysis

Notebook: `01_eda.ipynb`

### 1. Missing-Value Handling

| Column | Missing Values | Decision | Reason |
|---|---:|---|---|
| `age` | 19.87% | Median imputation | Retains rows and is less sensitive to outliers |
| `embarked`, `embark_town` | 0.22% | Drop two affected rows | Small amount of data loss |
| `deck` | 77.22% | Drop column | High missing-value proportion |

### 2. Fare Distribution

The fare distribution is right-skewed, with a small number of expensive tickets pulling the mean upward.

| Statistic | Fare |
|---|---:|
| Mean | 32.10 |
| Median | 14.45 |
| Mode | 8.05 |
| Approximate maximum | 512 |

### 3. Outlier Detection

Outliers were identified using the Interquartile Range (IQR) rule:

\[
\text{Lower Bound} = Q_1 - 1.5 \times IQR
\]

\[
\text{Upper Bound} = Q_3 + 1.5 \times IQR
\]

| Feature | Outliers Detected | Lower Bound | Upper Bound |
|---|---:|---:|---:|
| Age | 65 | 2.50 | 54.50 |
| Fare | 114 | -26.76 | 65.66 |

The detected outliers were retained because extreme values may represent genuine passenger ages or ticket prices rather than data errors. Tree-based models are generally less sensitive to extreme values than Logistic Regression.

### 4. Survival Analysis

#### Survival Rate by Sex and Passenger Class

| Group | Survival Rate |
|---|---:|
| Female passengers | 74.0% |
| Male passengers | 18.9% |
| First class | 62.6% |
| Second class | 47.3% |
| Third class | 24.2% |

#### Survival Rate by Sex and Class

| Sex | First Class | Second Class | Third Class |
|---|---:|---:|---:|
| Female | 96.7% | 92.1% | 50.0% |
| Male | 36.9% | 15.7% | 13.5% |

**Insights:**

- Female passengers had substantially higher survival rates than male passengers.
- Survival rates decreased from first class to third class.
- Third-class women had a lower survival rate than women in the first and second classes, but a higher rate than every male group shown.

### 5. Correlation Analysis

A correlation matrix and heatmap were generated for survival, passenger class, age, siblings/spouses, parents/children, and fare.

| Feature Pair | Correlation | Interpretation |
|---|---:|---|
| Fare and passenger class | -0.548 | Passenger class and fare were strongly associated |
| `parch` and `sibsp` | 0.415 | Family travel indicators were positively associated |
| Passenger class and survival | -0.336 | Passenger class was associated with survival |
| Fare and survival | 0.255 | Fare had a positive association with survival |

The analysis included survival-by-sex charts, survival-by-class charts, and a correlation heatmap.

---

## Stage 2: Machine Learning

Notebook: `02_modeling.ipynb`

### Modeling Workflow

```text
Load Titanic Dataset
        |
        v
Stratified 80/20 Train-Test Split
        |
        v
Preprocessing Pipeline
        |
        +-- Numeric Features
        |      Median Imputation
        |      Standard Scaling
        |
        +-- Categorical Features
               Most-Frequent Imputation
               One-Hot Encoding
        |
        v
Train Classification Models
        |
        +-- Logistic Regression
        +-- Decision Tree
        +-- Random Forest
        |
        v
Compare Models and Handle Class Imbalance
        |
        v
Hyperparameter Tuning with GridSearchCV
        |
        v
Evaluate Models and Save Pipeline
```

### Key Design Decisions

- **Stratified split:** Maintained similar survival-class proportions in the training and testing sets. The reported survival rates were 38.3% in training and 38.5% in testing.
- **Leakage prevention:** Imputation, encoding, and scaling were performed inside a scikit-learn pipeline fitted on training data.
- **Training-only resampling:** SMOTE was applied only to training data.
- **Feature selection:** Used `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, and `embarked`.
- **Excluded features:** `alive`, `adult_male`, `alone`, `who`, `class`, `embark_town`, and `deck` were excluded because they were duplicate, redundant, or mostly missing features.
- **Multiple evaluation metrics:** Precision, recall, F1 score, and ROC-AUC were used alongside accuracy.

---

## Model Evaluation

The following results were reported on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree (`max_depth=4`) | 0.793 | 0.864 | 0.551 | 0.673 | 0.829 |
| Random Forest — Default | **0.816** | 0.800 | 0.696 | 0.744 | 0.830 |
| Random Forest — Tuned | 0.799 | 0.789 | 0.652 | 0.714 | 0.840 |

Confusion matrices, ROC curves, and a labelled decision tree are included in the modeling notebook.

### Model Comparison

- The default Random Forest achieved the highest accuracy and F1 score among the listed models.
- Logistic Regression achieved the highest ROC-AUC.
- The tuned Random Forest did not outperform the default Random Forest on the held-out test set.

## Class Imbalance Handling

Different strategies were evaluated using Random Forest.

| Strategy | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Baseline | 0.800 | 0.696 | 0.744 |
| Class weighting (`balanced`) | 0.750 | 0.739 | 0.745 |
| SMOTE on training data | 0.761 | 0.739 | **0.750** |

Both class weighting and SMOTE increased survivor recall from 0.696 to 0.739, with lower precision.

SMOTE achieved the highest F1 score of 0.750, slightly exceeding the baseline score of 0.744. Since these differences represent a small number of predictions on a 179-passenger test set, further cross-validation is advisable.

## Hyperparameter Tuning

Random Forest hyperparameters were tuned using `GridSearchCV` with five-fold cross-validation and F1 scoring.

### Search Space

| Hyperparameter | Values Tested |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 3, 5, 7, `None` |
| `max_features` | `sqrt`, `log2`, `None` |

### Best Parameters

```python
{
    "clf__max_depth": 7,
    "clf__max_features": None,
    "clf__n_estimators": 300
}
```

| Metric | Result |
|---|---:|
| Best cross-validated F1 | 0.752 |
| Out-of-bag (OOB) score | 0.836 |
| Default Random Forest test F1 | 0.744 |
| Tuned Random Forest test F1 | 0.714 |

Although tuning produced a cross-validated F1 score of 0.752, the tuned model's test F1 decreased to 0.714. Based on the reported held-out test results, the default Random Forest remains the preferred candidate.

## Error Analysis

The reported Random Forest confusion matrix was:

|  | Predicted Not Survived | Predicted Survived |
|---|---:|---:|
| Actually Not Survived | 98 | 12 |
| Actually Survived | 21 | 48 |

The model correctly classified 98 non-survivors and 48 survivors. It incorrectly classified 12 non-survivors as survivors and missed 21 actual survivors.

- Approximately 30.4% of actual survivors were missed.
- Approximately 10.9% of actual non-survivors were incorrectly classified as survivors.

This suggests that identifying survivors was the more challenging task. Group-wise error rates by sex, passenger class, and age should be examined before drawing conclusions about which groups were most affected.

## Regression Side Task: Fare Prediction

A multivariate linear regression model was used to predict passenger fare.

| Metric | Reported Result |
|---|---:|
| MAE | 20.90 |
| RMSE | 30.53 |
| R² | 0.398 |
| Adjusted R² | 0.362 |

The reported residual analysis showed increasing error spread at higher predicted fares, suggesting heteroscedasticity. The reported correlation between predicted fare and absolute residual was 0.546.

The model explained approximately 39.8% of the variance in fare.

**Validation note:** The original project notes indicate that these regression metrics should be recalculated after removing `survived` from the fare-prediction predictors. Update the metrics and residual analysis after rerunning the notebook.

## Final Model Recommendation

Based on the current held-out test results, the default Random Forest is the preferred candidate.

It achieved:

- Accuracy: 0.816
- Precision: 0.800
- Recall: 0.696
- F1 score: 0.744
- ROC-AUC: 0.830

It outperformed Logistic Regression in accuracy, recall, and F1 score, although Logistic Regression achieved a higher ROC-AUC of 0.844.

The dataset is small, and the results rely on a single train-test split. Repeated cross-validation would provide a more reliable assessment before considering deployment.

---

## Using the Saved Pipeline

The fitted pipeline is saved as `titanic_full_pipeline.joblib`. It includes preprocessing and the trained model, allowing predictions from raw passenger inputs.

```python
import joblib
import pandas as pd

pipeline = joblib.load("titanic_full_pipeline.joblib")

passenger = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": None,
    "sibsp": 0,
    "parch": 0,
    "fare": 80.0,
    "embarked": "C"
}])

# Predict the passenger's survival class
prediction = pipeline.predict(passenger)

# Get predicted class probabilities
probabilities = pipeline.predict_proba(passenger)

print(prediction)
print(probabilities)
```

The example uses a first-class female passenger with an unspecified age, no listed siblings or spouse, no listed parents or children, a fare of 80.0, and embarkation code `C`.

---

## Installation and Usage

### Prerequisites

- Python installed on your system
- Git
- Jupyter Notebook

### 1. Clone the Repository

Replace the placeholders with your actual GitHub username and repository name.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebooks

1. Open `01_eda.ipynb` and run the exploratory data analysis.
2. Open `02_modeling.ipynb` and run the modeling workflow.

The `titanic.csv` dataset is included in the repository, so no separate dataset download is required.

### Dependencies

The `requirements.txt` file should include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
jupyter
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Preprocessing | Pipeline, ColumnTransformer |
| Hyperparameter Tuning | GridSearchCV |
| Class Imbalance | imbalanced-learn, SMOTE |
| Model Persistence | joblib |
| Development Environment | Jupyter Notebook |

## Limitations and Future Improvements

- **Limited dataset size:** The dataset contains 891 original rows, so results may vary with the train-test split. Repeated stratified cross-validation would provide a steadier performance estimate.
- **Feature engineering:** Evaluate features such as family size, fare per person, and age groups using the same cross-validation folds.
- **Error analysis:** Examine group-wise error rates and permutation feature importance to understand model behavior.
- **Alternative models:** Experiment with gradient boosting methods such as XGBoost or LightGBM.
- **Probability calibration:** Evaluate whether model probabilities are well calibrated.
- **Regression validation:** Recalculate fare-prediction metrics without using `survived` as a predictor.
- **Deployment:** Wrap the saved pipeline in a FastAPI endpoint to serve predictions through an API.

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow, from exploratory analysis and preprocessing to model comparison, hyperparameter tuning, error analysis, and pipeline persistence.

The default Random Forest achieved **81.6% accuracy** and an **F1 score of 0.744** on the reported held-out test set. The analysis also highlights how sex and passenger class were associated with survival outcomes.

The next steps are to strengthen evaluation through repeated cross-validation, validate the regression side task, and develop an API for serving predictions.
