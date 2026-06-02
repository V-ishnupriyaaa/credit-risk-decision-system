import pandas as pd

APPROVE_THRESHOLD = 0.35
REJECT_THRESHOLD = 0.55

def categorize_risk(probability):
    if probability < APPROVE_THRESHOLD:
        return 'APPROVE'
    elif probability <= REJECT_THRESHOLD:
        return 'REVIEW'
    else:
        return 'REJECT'

# preprocess_data()
def preprocess_data(df):
    df = df.copy()

     # Drop unnecessary column
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

     # Missing value treatment
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])

    # Outlier treatment
    df['DebtRatio'] = df['DebtRatio'].clip(upper=10)
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1.0)
    
    return df


# engineer_features()
def engineer_features(df):
    df = df.copy()
    
    # 1. Combined delinquency history
    df['total_delinquencies'] = df['NumberOfTime30-59DaysPastDueNotWorse'] + df['NumberOfTimes90DaysLate'] + df['NumberOfTime60-89DaysPastDueNotWorse']
    
    # 2. Drop original correlated columns
    df = df.drop(columns=['NumberOfTime30-59DaysPastDueNotWorse', 
                      'NumberOfTimes90DaysLate', 
                      'NumberOfTime60-89DaysPastDueNotWorse'])
    
    # 3. Utilization-Debt risk interaction
    df['utilization_debt_risk'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    
    # 4. Age-Income risk
    df['age_income_risk'] = (df['age']-60)*(12*df['MonthlyIncome'])
    
    return df
