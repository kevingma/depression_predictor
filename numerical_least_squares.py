import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv('./processed_data.csv')

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
required_cols = ['SEQN', 'depression_score']

for col in required_cols:
    if col not in numerical_cols:
        numerical_cols.append(col)

df_numerical = df[numerical_cols]

results = []

for column in df_numerical.columns:
    if column not in ['depression_score', 'SEQN']:
        col_data = df_numerical[column].fillna(df_numerical[column].mean())
        if col_data.nunique() > 1:
            X = col_data.values.reshape(-1, 1)
            y = df_numerical['depression_score'].values

            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            theta_best = np.linalg.lstsq(X_b, y, rcond=None)[0]
            y_pred = X_b.dot(theta_best)
            r2 = r2_score(y, y_pred)

            results.append({'Question': column, 'Coefficient': theta_best[1], 'Intercept': theta_best[0], 'R^2': r2})

results_df = pd.DataFrame(results)
results_df.to_csv('./least_squares_num_results.csv', index=False)
