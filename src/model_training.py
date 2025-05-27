import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_model(df):
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    # Split the dataset
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # ✅ Step 1: Create models/ folder if not exists
    os.makedirs("models", exist_ok=True)

    # ✅ Step 2: Save model
    joblib.dump(model, 'models/logistic_model.pkl')

    return model

if __name__ == "__main__":
    # Load and preprocess data
    from data_preprocessing import preprocess_data
    df = pd.read_csv("data/telco_churn.csv")
    df_cleaned = preprocess_data(df)
    train_model(df_cleaned)
