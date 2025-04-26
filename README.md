# Anxiety and Depression Dataset Analysis

This project helps to find the stress levels in individuals based on the factors such as physical activity, lack of meditation, financial stress etc.

# Overview
This project explores anxiety and depression levels influenced by modern lifestyle changes.
The dataset considers key factors such as reduced sleep, limited physical activity, lack of meditation, and financial stress.
As living styles and habits shift rapidly, mental health challenges are becoming increasingly common.

The objective of this analysis is to assess the impact of lifestyle factors on mental health, calculate individual stress levels, and predict the likelihood of anxiety or depression. The project also highlights the importance of maintaining a balance between work and personal well-being.

# Dataset Description

 Target Variable:

Stress_Level (Categorical: Levels of stress severity)

 Feature Variables:

Age

Gender

Education_Level

Employment_Status

Sleep_Hours

Physical_Activity_Hrs

Social_Support_Score

Anxiety_Score

Depression_Score

Family_History_Mental_Illness

Chronic_Illnesses

Medication_Use

Therapy

Meditation

Substance_Use

Financial_Stress

Work_Stress

Self_Esteem_Score

Life_Satisfaction_Score

Loneliness_Score

Anxiety_Level

These features include demographic details, health behaviors, psychological scores, and lifestyle factors that influence mental well-being.

# Packages Used

The following Python libraries and packages are used in this project:

Data Handling and Analysis:

pandas, numpy

Visualization:

matplotlib, seaborn

Statistical Analysis:

scipy

Machine Learning Models:

sklearn.ensemble (RandomForestClassifier)

sklearn.tree (DecisionTreeClassifier)

sklearn.neighbors (KNeighborsClassifier)

sklearn.svm (SVC)

sklearn.linear_model (LogisticRegression, LinearRegression)

sklearn.naive_bayes (MultinomialNB)

Preprocessing and Feature Engineering:

sklearn.preprocessing (StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler)

sklearn.impute (KNNImputer, SimpleImputer)

sklearn.compose (ColumnTransformer)

sklearn.pipeline (Pipeline)

Model Evaluation and Selection:

sklearn.model_selection (train_test_split, KFold, cross_val_score, GridSearchCV)

sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_squared_error, r2_score)

# Machine Learning Models Implemented

Logistic Regression (LogisticRegression(max_iter=1000))

Random Forest Classifier (RandomForestClassifier(max_depth=2, random_state=0))

Support Vector Machine (SVM) (SVC())

K-Nearest Neighbors (KNN) (KNeighborsClassifier())

Naive Bayes (MultinomialNB(alpha=0.15))

Decision Tree (Log-Odds Guided Custom Model) (LogOddsDecisionTree(max_depth=3))

Custom Model: Log-Odds Guided Decision Tree
A unique model named LogOddsDecisionTree was developed.
It builds decision trees using log-odds scores from logistic regression to guide feature splits, combining the interpretability of decision trees with the predictive strength of logistic regression.

# How to Run
Clone this repository.

Install the required packages by running:

pip install pandas numpy matplotlib seaborn scikit-learn scipy

Load the dataset and execute the notebook or Python scripts to train and evaluate the models.

# Evaluation Metrics

Models are evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Mean Squared Error (for regression tasks if any)

R² Score (for regression tasks if any)

# Conclusion

After performing extensive analysis and modeling, the following conclusions were drawn:

Logistic Regression achieved the highest accuracy of 98.6% in predicting stress levels, outperforming other models.

Logistic Regression models with feature importance analysis and with/without cross-validation produced identical results, demonstrating the robustness of the approach.

Cross-validation, regardless of fold size, consistently yielded the same accuracy levels. Thus, it did not significantly enhance model performance for this dataset.

Transforming the target variable (Stress_Level) from numeric to categorical significantly improved prediction metrics and overall model performance.

Therefore, Logistic Regression without cross-validation, when combined with feature importance selection on categorical target data, provides the best prediction results for identifying stress levels in individuals.

In summary, Logistic Regression is the most reliable model for this task, and a simple model without heavy cross-validation still achieves highly accurate and robust results.

# License

This project is open-source and available for educational and research purposes.

Took reference from kaggle to achieve the whole project.
