# Heart Disease Prediction

Binary classification model predicting presence or absence of heart disease from 13 clinical features, served as a containerised REST API.

Dataset: synthetic scale-up of the classic Cleveland Heart Disease dataset (630K training / 270K test rows).

### Data
https://www.kaggle.com/competitions/playground-series-s6e2/data

---

## Project Structure

```
├── api/
│   └── main.py              # FastAPI application
├── src/
│   ├── preprocess.py        # Feature encoding pipeline
│   ├── predict.py           # Inference helpers
│   └── train.py             # Retraining script
├── models/
│   └── model.pkl            # Trained XGBoost model
├── tests/
│   ├── test_preprocess.py   # Preprocessing unit tests
│   └── test_api.py          # API endpoint tests
├── data/
│   ├── train.csv            # 630,000 labelled rows
│   ├── test.csv             # 270,000 unlabelled rows
│   └── sample_submission.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-train.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the API

**With Docker (recommended):**

```bash
docker compose up --build
```

**Locally:**

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

---

## Making a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 55,
    "Sex": 1,
    "Chest pain type": 4,
    "BP": 130,
    "Cholesterol": 245,
    "FBS over 120": 0,
    "EKG results": 0,
    "Max HR": 150,
    "Exercise angina": 1,
    "ST depression": 1.2,
    "Slope of ST": 2,
    "Number of vessels fluro": 1,
    "Thallium": 7
  }'
```

**Response:**

```json
{
  "probability": 0.87,
  "prediction": "Presence"
}
```

---

## Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Retraining the Model

```bash
pip install -r requirements-train.txt
python -m src.train
```

Options:

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/train.csv` | Path to training CSV |
| `--output` | `models/model.pkl` | Path to save model |
| `--trials` | `10` | Number of Optuna trials |

Example:

```bash
python -m src.train --data data/train.csv --output models/model.pkl --trials 20
```

---

## Target

`Heart Disease`: `Presence` or `Absence`. Encoded as 1/0 in the submission file.

---

## Features

| Column | Type | Values | Description |
|---|---|---|---|
| Age | numerical | years | Patient age |
| Sex | binary | 1 = male, 0 = female | Biological sex |
| Chest pain type | categorical | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic | Type of chest pain reported |
| BP | numerical | mmHg | Resting blood pressure on admission |
| Cholesterol | numerical | mg/dl | Serum cholesterol level |
| FBS over 120 | binary | 1 = true, 0 = false | Fasting blood sugar > 120 mg/dl |
| EKG results | categorical | 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy | Resting electrocardiographic results |
| Max HR | numerical | bpm | Maximum heart rate achieved during exercise test |
| Exercise angina | binary | 1 = yes, 0 = no | Angina induced by exercise |
| ST depression | numerical | mm | ST depression induced by exercise relative to rest |
| Slope of ST | categorical | 1 = upsloping, 2 = flat, 3 = downsloping | Slope of peak exercise ST segment |
| Number of vessels fluro | numerical | 0–3 | Number of major vessels coloured by fluoroscopy |
| Thallium | categorical | 3 = normal, 6 = fixed defect, 7 = reversible defect | Thallium stress test result |
