import joblib
import pandas as pd
from src.preprocess import preprocess

MODEL_PATH = 'models/model.pkl'


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict(input_df: pd.DataFrame, model=None) -> list[float]:
    """
    Takes a raw DataFrame (as received from the API) and returns
    a list of heart disease probability scores.
    """
    if model is None:
        model = load_model()

    X = preprocess(input_df)
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities.tolist()
