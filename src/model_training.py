from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model(df):
    # Split features and target
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Prediction and metrics
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, './models/logistic_model.pkl')
    
    return model
