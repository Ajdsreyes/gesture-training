from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def train_model(X, y):

    # CHECK IF THERE ARE AT LEAST 2 USERS
    if len(set(y)) < 2:
        raise ValueError(
            "Need at least 2 different users/classes"
        )

    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # CREATE SVM MODEL
    model = SVC(kernel="rbf")

    # TRAIN MODEL
    model.fit(X_train, y_train)

    # TEST MODEL
    predictions = model.predict(X_test)

    # CALCULATE ACCURACY
    accuracy = accuracy_score(y_test, predictions)

    # SAVE MODEL
    joblib.dump(model, "svm_model.pkl")

    return {
        "accuracy": accuracy
    }


def evaluate_model(features):

    # LOAD SAVED MODEL
    model = joblib.load("svm_model.pkl")

    # PREDICT USER
    prediction = model.predict([features])

    return {
        "predicted_user": prediction[0]
    }