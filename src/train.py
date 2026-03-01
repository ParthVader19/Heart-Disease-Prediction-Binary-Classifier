"""
Train the heart disease XGBoost model with Optuna hyperparameter optimisation.

Usage:
    python -m src.train
    python -m src.train --data data/train.csv --output models/model.pkl --trials 10
"""
import argparse

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.preprocess import preprocess

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train(data_path: str, output_path: str, n_trials: int) -> None:
    # Load and prepare data
    train_df = pd.read_csv(data_path)
    train_df['Heart Disease'] = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    X = preprocess(train_df)
    y = train_df['Heart Disease']

    # Optuna objective
    def objective(trial):
        params = {
            'n_estimators': 500,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'enable_categorical': True,
            'tree_method': 'hist',
            'early_stopping_rounds': 50,
            'eval_metric': 'auc',
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        return np.mean(aucs)

    print(f"Running Optuna optimisation ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Best CV AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Train final model on full dataset
    print("Training final model...")
    final_model = XGBClassifier(
        **study.best_params,
        n_estimators=300,
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
    )
    final_model.fit(X, y)

    joblib.dump(final_model, output_path)
    print(f"Model saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train heart disease prediction model')
    parser.add_argument('--data',   default='data/train.csv',   help='Path to training CSV')
    parser.add_argument('--output', default='models/model.pkl', help='Path to save model')
    parser.add_argument('--trials', default=10, type=int,       help='Number of Optuna trials')
    args = parser.parse_args()

    train(args.data, args.output, args.trials)
