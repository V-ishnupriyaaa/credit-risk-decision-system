# imports
import joblib
import pandas as pd
import os
from utils import preprocess_data, engineer_features, categorize_risk

# RISK CATEGORIZATION
# Business-configurable thresholds — adjust based on risk appetite
# Narrowing REJECT_THRESHOLD reduces manual reviews but increases auto-rejections
# Current settings based on cost optimization (COST_FN:COST_FP = 10:1)



def run_inference():
    # Load saved model
    package = joblib.load('models/credit_risk_model.pkl')
    model = package['model']
    feature_names = package['features']
    
    # Load NEW applications
    df = pd.read_csv('data/new_applications.csv')
    print(df.columns.tolist())
    
    # Preprocess and engineer
    df_clean = preprocess_data(df)
    df_featured = engineer_features(df_clean)
    
    # Get features
    X = df_featured[feature_names]
    
    # Predict probabilities
    probabilities = model.predict_proba(X)[:, 1]
    
    # Build results
    results = pd.DataFrame({
        'default_probability': probabilities,
        'risk_decision': [categorize_risk(p) for p in probabilities]
    })
    
    os.makedirs('reports', exist_ok=True)
    results.to_csv('reports/final_decisions.csv', index=False)


    # Export
    results.to_csv('reports/final_decisions.csv', index=False)
    print(results['risk_decision'].value_counts())

if __name__ == "__main__":
    run_inference()