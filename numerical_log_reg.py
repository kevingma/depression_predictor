import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('./processed_data.csv')

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
required_cols = ['SEQN', 'depression_score']

for col in required_cols:
    if col not in numerical_cols:
        numerical_cols.append(col)

df_numerical = df[numerical_cols].copy()

threshold = df_numerical['depression_score'].median()
df_numerical.loc[:, 'depression_binary'] = (df_numerical['depression_score'] > threshold).astype(int)

results = []

for column in df_numerical.columns:
    if column not in ['depression_score', 'depression_binary', 'SEQN']:
        col_data = df_numerical[column].fillna(df_numerical[column].mean())

        if col_data.nunique() > 1:
            X = col_data.values.reshape(-1, 1)
            y = df_numerical['depression_binary'].values

            log_reg = LogisticRegression().fit(X, y)
            y_pred = log_reg.predict(X)

            accuracy = accuracy_score(y, y_pred)

            results.append({'Feature': column, 'Coefficient': log_reg.coef_[0][0], 'Intercept': log_reg.intercept_[0], 'Accuracy': accuracy})

results_df = pd.DataFrame(results)
results_df.to_csv('./log_reg_results.csv', index=False)
