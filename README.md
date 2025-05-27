# SaaS Churn Prediction

This project predicts customer churn for a SaaS business using machine learning. It includes data preprocessing, model training, inference, and a Streamlit web app for interactive predictions.

## Project Structure
- `data/`: Contains the raw dataset (`telco_churn.csv`).
- `models/`: Stores the trained model (`logistic_model.pkl`).
- `notebooks/`: Jupyter notebooks for EDA and inference demo.
- `src/`: Source code for preprocessing, training, inference, and the Streamlit app.
- `requirements.txt`: Python dependencies.

## Setup Instructions
1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Train the model (creates `models/logistic_model.pkl`):
   ```powershell
   python src/model_training.py
   ```
5. Run the Streamlit app:
   ```powershell
   streamlit run src/app.py
   ```

## Usage
- Use the Streamlit app to input customer features and get churn predictions interactively.
- See `notebooks/` for EDA and model inference examples.

## Example
![Streamlit Screenshot](reports/streamlit_screenshot.png)

---

**Author:** Your Name
**Date:** May 2025