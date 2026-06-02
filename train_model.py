# imports
import numpy as np
import pandas as pd 
import joblib
import os
from utils import preprocess_data, engineer_features
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


df = pd.read_csv('data/GiveMeSomeCredit-training.csv')
df_clean = preprocess_data(df)
df_featured = engineer_features(df_clean)


# training pipeline
X = df_featured.drop(columns=['SeriousDlqin2yrs'])
y = df_featured['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE 
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")


# Scale data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train_scaled, y_train_resampled)

# Model 2: Tuned Random Forest with embedded SMOTE 
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, scoring='f1', cv=5, n_jobs=-1)
grid_rf.fit(X_train_resampled, y_train_resampled) 
best_rf = grid_rf.best_estimator_

# Model 3: Tuned XGBoost with embedded SMOTE 
xgb_params = {
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3]
}
grid_xgb = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, scoring='f1', cv=5, n_jobs=-1)
grid_xgb.fit(X_train_resampled, y_train_resampled)
best_xgb = grid_xgb.best_estimator_


# --- Model Evaluation and Comparison ---
models_to_compare = [
    ("Logistic Regression", model_lr, X_test_scaled),
    ("Tuned Random Forest", best_rf, X_test),
    ("Tuned XGBoost",       best_xgb, X_test)
]

# Constants for Cost Optimization
COST_FN = 32000 # average loan loss from missed default
COST_FP = 3200 # lost interest from wrongly rejected customer
thresholds = np.arange(0.0, 1.0, 0.01)

best_cost = float('inf')  
best_model_name = ""
best_model_obj = None
best_threshold = 0.5
winning_f1 = 0

print("\n" + "="*40)
print("      COST-OPTIMIZED MODEL REPORT")
print("="*40)

for name, model, x_data in models_to_compare:
    y_probs = model.predict_proba(x_data)[:, 1]
    
    costs = []
    f1_scores = []

    for threshold in thresholds:
        y_pred_thresh = (y_probs >= threshold).astype(int) 
        cm = confusion_matrix(y_test, y_pred_thresh)
        
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fn * COST_FN) + (fp * COST_FP)
        current_f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        costs.append(total_cost)
        f1_scores.append(current_f1)

    min_cost_idx = np.argmin(costs)
    model_min_cost = costs[min_cost_idx]
    model_optimal_thresh = thresholds[min_cost_idx]
    model_f1_at_thresh = f1_scores[min_cost_idx]

    print(f"Model: {name}")
    print(f"  Min Cost:         {model_min_cost:,}")
    print(f"  Optimal Threshold: {model_optimal_thresh:.2f}")
    print(f"  f1 at Thresh:     {model_f1_at_thresh:.4f}")
    print("-" * 20)

    if model_min_cost < best_cost:
        best_cost = model_min_cost
        best_model_obj = model
        best_model_name = name
        best_threshold = model_optimal_thresh
        winning_f1 = model_f1_at_thresh

print(f"WINNING MODEL: {best_model_name}")
print(f"Optimal Threshold: {best_threshold:.2f}")
print(f"f1 at Threshold: {winning_f1:.4f}")
print(f"Minimum Total Cost: {best_cost:,}")

model_package = {
    'model': best_model_obj,
    'threshold': best_threshold,
    'scaler': scaler,
    'features': X.columns.tolist()
}

save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'models', 'credit_risk_model.pkl')
)

joblib.dump(model_package, save_path)
print("Model package saved successfully!")

X_test.head(10).to_csv('data/new_applications.csv', index=False)