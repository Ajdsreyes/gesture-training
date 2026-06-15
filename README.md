# Gesture Authentication Backend API

This project is a FastAPI backend for training and evaluating an SVM-based gesture authentication model.

The backend receives gesture feature sequences from the frontend data collection application, trains an SVM model, evaluates gesture sequences, and returns authentication results.

---

# Base URL

Local server:

```txt
http://127.0.0.1:5000
```

Swagger UI:

```txt
http://127.0.0.1:5000/docs
```

---

# Technologies Used

- FastAPI
- Python
- Scikit-learn
- SVM (Support Vector Machine)
- Joblib
- Uvicorn

---

# Project Structure

```txt
gesture-project/
│
├── main.py
├── model_training.py
├── svm_model.pkl
├── requirements.txt
└── README.md
```

---

# API Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | / | Server status |
| POST | /build-model | Train and save SVM model |
| POST | /evaluate | Evaluate gesture sequence |

---

# Route Details

---

# GET /

## Description

Checks if the backend server is running.

## Response

```json
{
  "message": "Gesture Authentication API Running"
}
```

---

# POST /build-model

## Description

Trains the SVM model using gesture sequence data and saves the trained model as:

```txt
svm_model.pkl
```

## Request Body

```json
{
  "X": [
    [0.1, 0.2, 0.3],
    [0.5, 0.6, 0.7],
    [0.9, 1.0, 1.1],
    [1.2, 1.3, 1.4]
  ],
  "y": [
    "user1",
    "user1",
    "user2",
    "user2"
  ]
}
```

## Requirements

- At least 2 different users/classes
- Multiple gesture samples per user

## Success Response

```json
{
  "message": "Model trained successfully",
  "accuracy": 0.95
}
```

---

# POST /evaluate

## Description

Evaluates a single gesture sequence using the trained SVM model.

## Request Body

```json
{
  "features": [0.2, 0.3, 0.4]
}
```

## Success Response

```json
{
  "predicted_user": "user1"
}
```

---

# How To Run

## 1. Install dependencies

```bash
pip install fastapi uvicorn scikit-learn joblib
```

---

## 2. Start the backend server

```bash
uvicorn main:app --reload --port 5000
```

---

## 3. Open Swagger UI

Go to:

```txt
http://127.0.0.1:5000/docs
```

---

# Training Workflow

1. Frontend collects gesture data
2. Frontend sends feature vectors to `/build-model`
3. Backend trains SVM model
4. Model saved as `svm_model.pkl`
5. Frontend sends new gesture sequence to `/evaluate`
6. Backend returns predicted user

---

# Notes

- The model must be trained first before evaluation.
- `svm_model.pkl` is automatically created after training.
- Evaluation will fail if the model file does not exist.
- At least 2 classes/users are required for SVM training.

---
