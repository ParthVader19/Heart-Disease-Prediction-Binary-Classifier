import pandas as pd

FEAT_NUM = [
    'Age',
    'BP',
    'Cholesterol',
    'Max HR',
    'ST depression',
    'Number of vessels fluro',
]

FEAT_CAT = [
    'Sex',
    'Chest pain type',
    'FBS over 120',
    'EKG results',
    'Exercise angina',
    'Slope of ST',
    'Thallium',
]

FEATURES = FEAT_NUM + FEAT_CAT

CAT_LEVELS = {
    'Sex':             [0, 1],
    'Chest pain type': [1, 2, 3, 4],
    'FBS over 120':    [0, 1],
    'EKG results':     [0, 1, 2],
    'Exercise angina': [0, 1],
    'Slope of ST':     [1, 2, 3],
    'Thallium':        [3, 6, 7],
}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply categorical encoding to a DataFrame. Works for both training and inference."""
    df = df.copy()
    for col, levels in CAT_LEVELS.items():
        df[col] = pd.Categorical(df[col].astype(int), categories=levels)
    return df[FEATURES]
