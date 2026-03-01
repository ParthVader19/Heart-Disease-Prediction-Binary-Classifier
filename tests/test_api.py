import pytest
from fastapi.testclient import TestClient
from api.main import app

SAMPLE_PAYLOAD = {
    'Age': 55,
    'BP': 130,
    'Cholesterol': 245,
    'Max HR': 150,
    'ST depression': 1.2,
    'Number of vessels fluro': 1,
    'Sex': 1,
    'Chest pain type': 4,
    'FBS over 120': 0,
    'EKG results': 0,
    'Exercise angina': 1,
    'Slope of ST': 2,
    'Thallium': 7,
}


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_response_schema(client):
    response = client.post('/predict', json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 'probability' in body
    assert 'prediction' in body
    assert isinstance(body['probability'], float)
    assert 0.0 <= body['probability'] <= 1.0
    assert body['prediction'] in ('Presence', 'Absence')


def test_invalid_age(client):
    payload = {**SAMPLE_PAYLOAD, 'Age': 999}
    response = client.post('/predict', json=payload)
    assert response.status_code == 422


def test_missing_field(client):
    payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != 'Thallium'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 422
