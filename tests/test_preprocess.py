import pandas as pd
import pytest
from src.preprocess import preprocess, FEATURES, CAT_LEVELS


SAMPLE_ROW = {
    'Age': 55, 'BP': 130, 'Cholesterol': 245, 'Max HR': 150,
    'ST depression': 1.2, 'Number of vessels fluro': 1,
    'Sex': 1, 'Chest pain type': 4, 'FBS over 120': 0,
    'EKG results': 0, 'Exercise angina': 1,
    'Slope of ST': 2, 'Thallium': 7,
}


def test_output_columns():
    df = preprocess(pd.DataFrame([SAMPLE_ROW]))
    assert list(df.columns) == FEATURES


def test_category_levels():
    df = preprocess(pd.DataFrame([SAMPLE_ROW]))
    for col, levels in CAT_LEVELS.items():
        assert list(df[col].cat.categories) == levels


def test_unseen_category_value():
    row = SAMPLE_ROW.copy()
    row['Thallium'] = 99  # value not in training levels
    df = preprocess(pd.DataFrame([row]))
    assert pd.isna(df['Thallium'].iloc[0])  # should be NaN, not an error
