import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            r2 = r2_score(y, y_pred)

            results.append({'Question': column, 'Coefficient': reg.coef_[0], 'Intercept': reg.intercept_, 'R^2': r2})
results_df = pd.DataFrame(results)
results_df.to_csv('./lin_reg_num_results.csv', index=False)
