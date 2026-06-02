import shap
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import preprocess_data, engineer_features, categorize_risk
import os

# Load saved model package
package = joblib.load('models/credit_risk_model.pkl')
model = package['model']
feature_names = package['features']

# Load and preprocess data
df = pd.read_csv('data/GiveMeSomeCredit-training.csv')
df_clean = preprocess_data(df)
df_featured = engineer_features(df_clean)

X = df_featured.drop(columns=['SeriousDlqin2yrs'])
y = df_featured['SeriousDlqin2yrs']

# Note: In production, X_test would be real new applications
# Using same random_state=42 to reproduce identical test split from training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SHAP explanation
X_test_sample = X_test.sample(1000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

print(type(shap_values))
print("Length:", len(shap_values))
if isinstance(shap_values, list):
    print("List item 0 shape:", shap_values[0].shape)
    print("List item 1 shape:", shap_values[1].shape)
else:
    print("Array shape:", shap_values.shape)

# Force regular SHAP values, not interaction values
shap_values = explainer.shap_values(X_test_sample, check_additivity=False)

# For 3D array — take mean across interaction dimension
if len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]  
else:
    sv = shap_values

shap.summary_plot(sv, X_test_sample, feature_names=feature_names)

# Show explanation for one individual applicant
i = 0  # first applicant in sample

shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value[1],
    sv[i],
    feature_names=feature_names
)

# Top risk factor per applicant
top_factors = [feature_names[np.argmax(np.abs(sv[i]))] 
               for i in range(len(X_test_sample))]

# Get probabilities
probs = model.predict_proba(X_test_sample)[:, 1]

# Build explanation dataframe
explanation_df = pd.DataFrame({
    'default_probability': probs,
    'risk_decision': [categorize_risk(p) for p in probs],
    'top_risk_factor': top_factors
})

print(explanation_df.head(10))
explanation_df.to_csv('reports/explanations.csv', index=False)
print("Explanations saved successfully!")