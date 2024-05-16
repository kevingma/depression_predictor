import pandas as pd
import numpy as np

df = pd.read_csv('./processed_data.csv', header=0)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
required_cols = ['SEQN', 'depression_score']

for col in required_cols:
    if col not in numerical_cols:
        numerical_cols.append(col)

df_numerical = df[numerical_cols]

correlation_dict = {}

for column in df_numerical.columns:
    if column not in ['depression_score', 'SEQN']:
        col_data = df_numerical[column].fillna(df_numerical[column].mean())

        if col_data.nunique() > 1:
            correlation = np.corrcoef(col_data, df_numerical['depression_score'])[0, 1]
            correlation_dict[column] = correlation
        else:
            x = 0

correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['Feature', 'Correlation'])

correlation_df['Abs_Correlation'] = correlation_df['Correlation'].abs()
correlation_df = correlation_df.sort_values(by='Abs_Correlation', ascending=False).drop(columns=['Abs_Correlation'])
correlation_df.to_csv('./pearson_results.csv', index=False)