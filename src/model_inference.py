import joblib
import pandas as pd

def load_model(path='models/logistic_model.pkl'):
    return joblib.load(path)

def predict_new(model, input_df):
    return model.predict(input_df)
