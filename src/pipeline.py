import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from joblib import dump

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    logging.info(f"Loaded data with shape: {data.shape}")
    data.fillna(data.mode().iloc[0], inplace=True)
    return data

# Split early to avoid leakage
def split_data(data, target='default'):
    X = data.drop(columns=[target])
    y = data[target]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature engineering
def feature_engineering(df):
    df = df.copy()
    
    df['income_level'] = pd.cut(df['monthly_income'], bins=[0, 5000, 10000, 15000, 20000, np.inf],
                                labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'])
    df['payment_behavior'] = df['utility_payment_timeliness'].apply(lambda x: 'Good' if x in ['early', 'on-time'] else 'Bad')
    df['loan_to_income_ratio'] = df['loan_amount'] / df['monthly_income']
    df['payment_discipline_score'] = df['utility_payment_timeliness'].map({'early': 1, 'on-time': 0.5, 'late': 0})

    income_level_map = {'Low': 1.0, 'Lower-Middle': 0.8, 'Middle': 0.6, 'Upper-Middle': 0.4, 'High': 0.2}
    df['income_adjusted_loan_ratio'] = df['loan_to_income_ratio'] / df['income_level'].map(income_level_map).astype(float)
    df['reliable_borrower'] = ((df['has_previous_loan'] == 1) & (df['payment_behavior'] == 'Good')).astype(int)

    return df

# Preprocessing pipeline
def build_preprocessor():
    numeric_features = [
        'monthly_income', 'loan_to_income_ratio', 'payment_discipline_score',
        'income_adjusted_loan_ratio'
    ]
    categorical_features = ['gender', 'utility_payment_timeliness', 'income_level', 'payment_behavior']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return preprocessor

# Build and train model
def build_train_pipeline(X_train, y_train, X_test, y_test):
    preprocessor = build_preprocessor()
    smote = SMOTE(random_state=42)
    pca = PCA(n_components=0.95)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('pca', pca),
        ('model', xgb_model)
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.01, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }

    search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10,
                                scoring='f1', cv=3, verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    path = './Microloan Default Prediction/saved_models/loan_default_pipeline.joblib'
    if not os.path.exists('./Microloan Default Prediction/saved_models'):
        os.makedirs('./saved_models')
    dump(best_model, './Microloan Default Prediction/saved_models/loan_default_pipeline.joblib')
    logging.info("Model saved as 'loan_default_pipeline.joblib'")

    # Evaluate
    y_pred = best_model.predict(X_test)
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logging.info(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Save the evaluation metrics
    with open('./Microloan Default Prediction/evaluation_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}\n")
        f.write(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}\n")
        f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
        f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")


    return best_model

# Main entry
def main():
    file_path = './Microloan Default Prediction/datasets/simulated_loan_data.csv'
    raw_data = load_data(file_path)

    # Split first
    X_train, X_test, y_train, y_test = split_data(raw_data)

    # Apply feature engineering on both sets
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    model = build_train_pipeline(X_train, y_train, X_test, y_test)
    logging.info("Training complete.")

if __name__ == "__main__":
    main()