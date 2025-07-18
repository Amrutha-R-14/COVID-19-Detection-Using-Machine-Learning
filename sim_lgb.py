import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, f1_score
import shap
import lightgbm as lgb

# Load the dataset
file_path = 'C://Users//AMRUTHA R//Downloads//cov_full1.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path, low_memory=False)
# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

# Display the first 20 rows of the DataFrame with all columns
print(data.head(20))

# Basic Data Preprocessing
data['age_60_and_above'].fillna(data['age_60_and_above'].mode()[0], inplace=True)
data['gender'].fillna(data['gender'].mode()[0], inplace=True)

# Filter to keep only binary classes (positive vs. negative)
binary_classes = ['positive', 'negative']
data = data[data['corona_result'].isin(binary_classes)]
print(data.size)
# Encode categorical variables
le = LabelEncoder()
data['corona_result'] = le.fit_transform(data['corona_result'])
data['age_60_and_above'] = le.fit_transform(data['age_60_and_above'])
data['gender'] = le.fit_transform(data['gender'])
data['test_indication'] = le.fit_transform(data['test_indication'])

# Feature Selection (Dropping 'test_date' as it's not relevant for modeling)
X = data.drop(columns=['corona_result', 'test_date'])
y = data['corona_result']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define LightGBM model with hyperparameters and perform Grid Search
params = {
    'num_leaves': [20, 30],
    'min_data_in_leaf': [4, 10],
    'feature_fraction': [0.2, 0.4],
    'bagging_fraction': [0.8, 0.9],
    'bagging_freq': [5, 10],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(estimator=lgb.LGBMClassifier(objective='binary', verbose=-1), param_grid=params, cv=3, scoring='f1')
grid.fit(X_train, y_train)

# Best model from Grid Search
model = grid.best_estimator_

# Predict on test data
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
auprc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve (area = {auprc:.2f})')
plt.show()

# Confusion Matrix with Class Labels
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['negative', 'positive'], columns=['negative', 'positive'])
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['negative', 'positive'], zero_division=1))

# SHAP Values
# SHAP Values
# SHAP Values with feature names and labels for positive/negative impact
# SHAP Values with feature names
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Beeswarm Plot with feature names
    feature_names = X.columns  # Assuming X is your original dataframe before scaling
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="dot", show=False)
    
    plt.title('SHAP Beeswarm Plot for COVID-19 Diagnosis Prediction')
    plt.xlabel('SHAP value (impact on model output)')
    plt.ylabel('Features')
    plt.show()

except Exception as e:
    print(f"Error in SHAP analysis: {e}")

import seaborn as sns

importance = model.booster_.feature_importance(importance_type='gain')
feature_names = X.columns  # Use the same feature names as in the SHAP plot

# Create a DataFrame for easy manipulation
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)

# Plot using seaborn with feature names on y-axis
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = round(accuracy_score(y_test, y_pred),4)

# Print the accuracy
print(f'Accuracy: {accuracy}')
