Titanic Survival Analysis & Prediction: EDA to Deployable ML Pipeline
An end-to-end data science project on the Titanic passenger dataset. It
has two stages: exploratory data analysis (EDA), which cleans the data
and identifies survival patterns, followed by a modeling stage that
builds, compares, and tunes classifiers and saves a pipeline that
predicts from raw input.

Headline results: Survival was strongly associated with sex and
passenger class: 74.0% of women survived compared with 18.9% of men;
survival was 62.6% in first class compared with 24.2% in third class.
The default Random Forest reached 81.6% accuracy, 0.744 F1, and 0.830
ROC-AUC on a held-out test set of 179 passengers.

Built as part of my AI/ML capstone project.
Repository Structure
.
├── titanic.csv                   # raw dataset (offline copy saved by the EDA notebook)
├── 01_eda.ipynb                  # Stage 1: cleaning and exploratory analysis
├── 02_modeling.ipynb             # Stage 2: modeling, tuning, comparison, saved pipeline
├── titanic_full_pipeline.joblib  # saved end-to-end model pipeline
├── requirements.txt
└── README.md
Make sure the notebook filenames above match the filenames actually
committed to the repository.

1. Problem Definition
Goal: Predict whether a passenger survived (survived = 1) from
passenger class, sex, age, family size, fare, and port of embarkation,
and understand which factors are associated with survival.

Secondary task: Predict fare from other passenger attributes using
multivariate linear regression and diagnose the model with residual
analysis.
2. Dataset
  Item                                Detail
  Source                              Titanic dataset loaded with
                                      seaborn.load_dataset("titanic")
                                      and saved as titanic.csv

  Raw size                            891 rows, 15 columns

  After EDA cleaning                  889 rows, 14 columns

  Target                              survived (0 = no, 1 = yes)
  Survival rate                       Approximately 38.4%
3. Stage 1: Exploratory Data Analysis (01_eda.ipynb)
Missing-value handling
  Column                         Missing Decision         Reason
  age                           19.87% Imputed with the Median
                                         median           imputation
                                                          retains rows and
                                                          is less
                                                          sensitive to
                                                          outliers

  embarked,                      0.22% Dropped the two  The amount of
  embark_town                          affected rows    data loss is
                                                          small

  deck                          77.22% Dropped the      The high missing
                                         column           proportion makes
                                                          imputation
                                                      unreliable
Distribution of fare
Fare is right-skewed: the mean was 32.10, the median was 14.45, and the
mode was 8.05. A small number of expensive tickets (up to approximately
512) pull the mean upward.
Outlier detection (IQR rule)
Using the IQR rule (1.5 × IQR), age has 65 outliers (bounds 2.50
to 54.50) and fare has 114 outliers (bounds -26.76 to 65.66).
These observations were retained because extreme values can represent
genuine passengers and ticket prices rather than data errors. Outliers
were detected but not removed; tree-based models are generally less
sensitive to extreme values than Logistic Regression.
Survival patterns
  Group         Survival rate
  Female                74.0%
  Male                  18.9%
  1st class             62.6%
  2nd class             47.3%
  3rd class             24.2%

Survival by sex and class

  Sex        1st class   2nd class   3rd class
  Female         96.7%       92.1%       50.0%
  Male           36.9%       15.7%       13.5%

Sex was the strongest single factor in the reported survival patterns,
and passenger class also mattered within each sex. Third-class women
(50.0%) survived at a lower rate than first- and second-class women, but
above every group of men, including first-class men (36.9%).
Correlation analysis
A correlation matrix and heatmap were computed for survived, pclass,
age, sibsp, parch, and fare.
- fare and pclass had the strongest reported pairwise correlation
  (-0.548): lower class numbers correspond to higher passenger classes
  and generally higher fares.
- parch and sibsp had a correlation of 0.415, indicating that
  passengers travelling with siblings or a spouse also often travelled
  with parents or children.
- Against survival, pclass (-0.336) and fare (0.255) had the
  strongest reported linear relationships.
Multivariate charts
- Survival by sex: Female passengers had a substantially higher
  survival rate than male passengers.
- Survival by passenger class: Survival decreased from first class
  to third class.
- Correlation heatmap: Fare and passenger class had the strongest
  reported pairwise correlation (-0.548).
4. Stage 2: Modeling (02_modeling.ipynb)
titanic.csv
   |
   v
Stratified 80/20 train-test split (before preprocessing)
   |
   v
ColumnTransformer inside a Pipeline
   |-- numeric: median imputation -> StandardScaler
   '-- categorical: most-frequent imputation -> one-hot encoding
   |
   v
Logistic Regression | Decision Tree | Random Forest
   |
   v
Imbalance comparison -> GridSearchCV tuning -> final comparison -> saved pipeline
Key design decisions
- Stratified split: The split was stratified on survived so that
  training and test sets retained similar class proportions. The
  reported survival rates were 38.3% in training and 38.5% in
  testing.
- No data leakage: The EDA cleaning is used for exploration. The
  modeling stage starts from titanic.csv and performs imputation,
  encoding, and scaling within a scikit-learn pipeline fitted on the
  training split. SMOTE is applied only to training data.
- Features used: pclass, sex, age, sibsp, parch, fare,
  and embarked. alive (a copy of the target), adult_male,
  alone, who, class, embark_town, and deck were excluded as
  duplicate, redundant, or mostly missing features.
- Multiple metrics: Precision, recall, F1, and ROC-AUC supplement
  accuracy and help evaluate performance on the minority survivor
  class.
4.1 Classifier comparison (held-out test set)
  Model                 Accuracy    Precision       Recall           F1      ROC-AUC
  Logistic                 0.804        0.793        0.667        0.724        0.844
  Regression                                                            

  Decision Tree            0.793        0.864        0.551        0.673        0.829
  (max_depth=4)                                                       

  Random Forest            0.816        0.800        0.696        0.744        0.830
  Random Forest            0.799        0.789        0.652        0.714        0.840
  (tuned)                                                               
Confusion matrices, ROC curves, and the labelled decision tree are
included in the notebook.
4.2 Class-imbalance handling (Random Forest)
  Strategy                       Precision   Recall      F1
  Baseline                           0.800    0.696   0.744
  class_weight='balanced'          0.750    0.739   0.745
  SMOTE (training data only)         0.761    0.739   0.750

Conclusion: Both class weighting and SMOTE increased survivor recall
from 0.696 to 0.739, at the cost of lower precision. SMOTE achieved the
highest F1 score (0.750), narrowly outperforming the baseline (0.744).
Because the differences represent only a small number of predictions on
the 179-passenger test set, further cross-validation is advisable.
4.3 Hyperparameter tuning
- Method: GridSearchCV, 5-fold cross-validation, scoring on F1.
- Search space: n_estimators in {100, 200, 300}, max_depth
  in {3, 5, 7, None}, and max_features in {sqrt, log2, None}.
- Best parameters:
  {'clf__max_depth': 7, 'clf__max_features': None, 'clf__n_estimators': 300}
- Best cross-validated F1: 0.752
- Out-of-bag (OOB) score: 0.836
- Effect of tuning: Test F1 decreased from 0.744 to 0.714. The
  tuned model did not outperform the default Random Forest on the
  held-out test set, so the default model remains the preferred
  candidate based on these results.
4.4 Error analysis
The previously reported Random Forest confusion matrix was:
                        Predicted not survived   Predicted survived
  Actually not survived                         98                   12
  Actually survived                             21                   48

The Random Forest correctly classified 98 non-survivors and 48
survivors. It incorrectly classified 12 non-survivors as survivors and
missed 21 actual survivors. It missed approximately 30.4% of survivors,
compared with a false-positive rate of approximately 10.9% among
non-survivors. This indicates that identifying survivors was the more
challenging task. Group-wise error rates by sex, passenger class, and
age should be checked in the notebook before making claims about which
groups are most affected.
4.5 Regression side-task: predicting fare
  MAE    RMSE      R²   Adjusted R²
20.90   30.53   0.398         0.362
The previously reported residual analysis indicated increasing error
spread at higher predicted fares, with a correlation of 0.546 between
predicted fare and absolute residual. This suggests heteroscedasticity,
meaning the error variance is not constant. The model explained
approximately 39.8% of the variance in fare. A log transformation of
fare or a different regression model could be explored to improve
performance.
Recalculate these regression metrics if survived is removed from the
regression predictors, as it should not be used to predict ticket
fare. Update the values and residual description to match the rerun
notebook output.

4.6 Final recommendation
I would select the default Random Forest as the preferred model based on
the current held-out test results. It achieved 0.816 accuracy, 0.800
precision, 0.696 recall, 0.744 F1, and 0.830 ROC-AUC. It outperformed
Logistic Regression in accuracy, recall, and F1, although Logistic
Regression achieved a higher ROC-AUC of 0.844. The tuned Random Forest
achieved a lower test F1 of 0.714, so tuning did not improve the result
on this split. The main limitation is the small dataset and reliance on
a single train-test split; repeated cross-validation would provide a
more reliable assessment before deployment.
5. Using the Saved Pipeline
The fitted pipeline (imputer, encoder, scaler, and final model) is saved
as titanic_full_pipeline.joblib and can predict directly from raw
data:
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
    "embarked": "C",
}])

print(pipeline.predict(passenger))       # predicted class
print(pipeline.predict_proba(passenger)) # class probabilities
6. How to Run
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]

python -m venv .venv
Activate the environment on Windows:
.venv\Scripts\activate
Install dependencies and start Jupyter:
pip install -r requirements.txt
jupyter notebook
Run the EDA notebook first, then the modeling notebook. titanic.csv is
included, so no dataset download is required.
requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
jupyter
7. Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn (Pipeline,
ColumnTransformer, GridSearchCV), imbalanced-learn (SMOTE), joblib,
and Jupyter Notebook.
8. Limitations and Next Steps
- The dataset is small (891 rows), so test metrics can vary with the
  random split. Repeated stratified cross-validation would provide a
  steadier estimate.
- Engineered features such as family size, fare per person, and age
  groups could improve performance and should be evaluated against the
  baseline using the same cross-validation folds.
- Group-wise error analysis and permutation feature importance can
  help explain where the model makes mistakes and which input features
  contribute most to predictive performance.
- Gradient boosting (such as XGBoost or LightGBM) and probability
  calibration are possible next experiments.
- The fitted model is saved as a file. Wrapping it in a FastAPI
  endpoint would be a next step toward serving predictions through an
  API.
