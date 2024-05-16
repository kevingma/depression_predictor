import pandas as pd
import numpy as np
import keras
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

model = keras.saving.load_model('my_model.keras')

scaler = joblib.load('scaler.pkl')

df_test = pd.read_csv('processed_data.csv')

numerical_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
required_cols = ['SEQN', 'depression_score']

for col in required_cols:
    if col not in numerical_cols:
        numerical_cols.append(col)

df_numerical = df_test[numerical_cols].copy()

df_numerical = df_numerical.fillna(df_numerical.mean())

X_test = df_numerical.drop(columns=['depression_score', 'SEQN'])
y_test = df_numerical['depression_score']

X_test = scaler.transform(X_test)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MSE: {mse}')
print(f'Test MAE: {mae}')

results_df = pd.DataFrame({'True Value': y_test, 'Predicted Value': y_pred.flatten()})
print(results_df.head())
