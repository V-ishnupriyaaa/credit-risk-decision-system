import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
from utils import preprocess_data, engineer_features
import matplotlib.pyplot as plt

# Load model
package = joblib.load('models/credit_risk_model.pkl')
model = package['model']
feature_names = package['features']

# Load and prepare data
df = pd.read_csv('data/GiveMeSomeCredit-training.csv')
df_clean = preprocess_data(df)
df_featured = engineer_features(df_clean)

X = df_featured.drop(columns=['SeriousDlqin2yrs'])
y = df_featured['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get probabilities
y_probs_final = model.predict_proba(X_test)[:, 1]

# simulate_strategy function here
def simulate_strategy(strategy_name, approve_threshold, y_probs, y_test):
    COST_FN = 32000
    COST_FP = 3200
    
    y_pred = (y_probs >= approve_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    loss_from_defaults = fn * COST_FN
    loss_from_rejections = fp * COST_FP
    total_loss = loss_from_defaults + loss_from_rejections
    
    print(f"Strategy: {strategy_name}")
    print(f"  Threshold: {approve_threshold}")
    print(f"  Missed Defaulters (FN): {fn}")
    print(f"  Wrongly Rejected (FP): {fp}")
    print(f"  Loss from Defaults: ${loss_from_defaults:,}")
    print(f"  Loss from Rejections: ${loss_from_rejections:,}")
    print(f"  Total Loss: ${total_loss:,}")
    print("-" * 30)

# Three strategies
simulate_strategy("Conservative", 0.25, y_probs_final, y_test)
simulate_strategy("Balanced",     0.35, y_probs_final, y_test)
simulate_strategy("Aggressive",   0.50, y_probs_final, y_test)

# Visualiziation

strategies = ['Conservative\n(0.25)', 'Balanced\n(0.35)', 'Aggressive\n(0.50)']
default_losses = [12512000, 18016000, 26880000]
rejection_losses = [24988800, 17321600, 9648000]
total_losses = [37500800, 35337600, 36528000]

x = range(len(strategies))

plt.figure(figsize=(10, 6))
plt.bar(x, default_losses, label='Loss from Defaults (FN)', color='red', alpha=0.7)
plt.bar(x, rejection_losses, bottom=default_losses, label='Loss from Rejections (FP)', color='orange', alpha=0.7)
plt.xticks(x, strategies)
plt.ylabel('Total Financial Loss ($)')
plt.title('Financial Impact by Threshold Strategy')
plt.legend()
plt.tight_layout()
plt.savefig('reports/strategy_comparison.png')
plt.show()
print("Chart saved to reports/strategy_comparison.png")