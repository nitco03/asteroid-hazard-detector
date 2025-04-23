# neo_classifier.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("dataset/neo.csv")

# Drop irrelevant columns
df = df.drop(['id', 'name', 'orbiting_body', 'sentry_object'], axis=1)

# Convert target to integer
df['hazardous'] = df['hazardous'].astype(int)

# Separate majority and minority classes
df_majority = df[df['hazardous'] == 0]
df_minority = df[df['hazardous'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Prepare train/test split
X = df_balanced.drop('hazardous', axis=1)
y = df_balanced['hazardous']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weight using original imbalance (realistic)
orig_df = pd.read_csv("dataset/neo.csv")
n_hazardous = len(orig_df[orig_df['hazardous'] == 1])
n_non_hazardous = len(orig_df[orig_df['hazardous'] == 0])
scale_pos_weight = n_non_hazardous / n_hazardous

# Train XGBoost with real-world class imbalance awareness
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_pred = xgb_model.predict(X_test)

# Evaluation
print("Confusion Matrix (XGBoost):")
print(confusion_matrix(y_test, xgb_pred))
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_pred))

# Save model
joblib.dump(xgb_model, 'xgboost_asteroid_model.pkl')
print("Model saved!")

# Feature importance
xgb_feat_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_feat_importances.nlargest(10).plot(kind='barh', title='XGBoost Feature Importances')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Real class distribution
print("Original Dataset Class Distribution:")
print(orig_df['hazardous'].value_counts())
