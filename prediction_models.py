import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
import joblib
from keras import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('./processed_data.csv')

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
required_cols = ['SEQN', 'depression_score']

for col in required_cols:
    if col not in numerical_cols:
        numerical_cols.append(col)

# Include only numerical response questions, replace empty answers with mean
df_numerical = df[numerical_cols].copy()
df_numerical = df_numerical.dropna(axis=1, how='all')
imputer = SimpleImputer(strategy='mean')
df_numerical_imputed = pd.DataFrame(imputer.fit_transform(df_numerical), columns=df_numerical.columns)

X = df_numerical_imputed.drop(columns=['depression_score', 'SEQN'])
y = df_numerical_imputed['depression_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Verify shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Mean of actual values in test set
mean_actual = np.mean(y_test)

# Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=83)
rf.fit(X_train, y_train)

# Predict and test
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
perc_err_rf = (mae_rf / mean_actual) * 100
print(f'Random Forest - Test MSE: {mse_rf}, Test MAE: {mae_rf}, Test Percentage Error: {perc_err_rf}%')

joblib.dump(rf, 'rf_model.pkl')

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=83)
xgb_model.fit(X_train, y_train)

# Predict and test
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
perc_err_xgb = (mae_xgb / mean_actual) * 100
print(f'XGBoost - Test MSE: {mse_xgb}, Test MAE: {mae_xgb}, Test Percentage Error: {perc_err_xgb}%')

joblib.dump(xgb_model, 'xgb_model.pkl')

# Create neural network
model = Sequential()
model.add(layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
y_pred_nn = model.predict(X_test)

# Evaluate the model
mse_nn = mean_squared_error(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
perc_err_nn = (mae_nn / mean_actual) * 100
print(f'Neural Network - Test MSE: {mse_nn}, Test MAE: {mae_nn}, Test Percentage Error: {perc_err_nn}%')

model.save('nn_model.keras')
joblib.dump(scaler, 'scaler.pkl')

models = ['Random Forest', 'XGBoost', 'Neural Network']
mse_values = [mse_rf, mse_xgb, mse_nn]
mae_values = [mae_rf, mae_xgb, mae_nn]
perc_err_values = [perc_err_rf, perc_err_xgb, perc_err_nn]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title('Model Performance Comparison')
ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Absolute Error (MAE)', color='tab:blue')
ax1.bar(models, mae_values, color='tab:blue', alpha=0.6, label='MAE')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Percentage Error (%)', color='tab:red')
ax2.plot(models, perc_err_values, color='tab:red', marker='o', label='Percentage Error')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
plt.savefig('./perf_visualizer.png')
plt.show()
