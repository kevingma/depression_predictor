import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from transformers import BertTokenizer, BertModel
import torch

# Encodes text using BERT
def berter(texts, model, tokenizer):
    encoded_texts = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        encoded_texts.append(embeddings)
    return np.array(encoded_texts)

df = pd.read_csv('./processed_data.csv')

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

r2_scores = []

# linreg
for column in df.columns:
    if column not in ['depression_score', 'SEQN']:
        if df[column].dtype == object:
            # Encode text data using BERT
            texts = df[column].fillna("").tolist()  # Replace NaNs with empty strings
            encoded_texts = berter(texts, model, tokenizer)
            col_data = encoded_texts
        else:
            col_data = df[column].fillna(df[column].mean()).values.reshape(-1, 1)

        depression_score = df['depression_score'].values

        if np.isnan(col_data).any() or np.isnan(depression_score).any():
            continue

        # Fit
        lin_model = LinearRegression()
        lin_model.fit(col_data, depression_score)

        # Predict and calculate R^2
        predictions = lin_model.predict(col_data)
        r2 = r2_score(depression_score, predictions)

        r2_scores.append((column, r2))

r2_df = pd.DataFrame(r2_scores, columns=['Feature', 'R2_Score'])
r2_df = r2_df.sort_values(by='R2_Score', ascending=False)

r2_df.to_csv('./non_numerical_lin_reg.csv', index=False)
