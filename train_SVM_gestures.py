import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import joblib

# ==========================================
# 1. FEATURE EXTRACTION FROM RAW GESTURE
# ==========================================

def extract_features(gesture):
    """
    gesture = list of points
    each point:
    {
        "time": float,
        "x": float,
        "y": float,
        "pressure": float,
        "fingers": [(x1,y1), (x2,y2)] OPTIONAL
    }
    """

    if len(gesture) < 2:
        return [0]*15

    times = np.array([p["time"] for p in gesture])
    xs = np.array([p["x"] for p in gesture])
    ys = np.array([p["y"] for p in gesture])
    pressures = np.array([p["pressure"] for p in gesture])

    # ---------------- CORE FEATURES ----------------
    duration = times[-1] - times[0]
    start_x, start_y = xs[0], ys[0]
    end_x, end_y = xs[-1], ys[-1]

    mean_pressure = np.mean(pressures)
    pressure_std = np.std(pressures)

    displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

    # ---------------- DYNAMIC FEATURES ----------------
    speeds = []
    for i in range(1, len(gesture)):
        dt = times[i] - times[i-1]
        if dt == 0:
            continue
        dist = np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
        speeds.append(dist / dt)

    speeds = np.array(speeds) if len(speeds) > 0 else np.array([0])

    mean_speed = np.mean(speeds)
    speed_std = np.std(speeds)

    slope = (end_y - start_y) / (end_x - start_x + 1e-6)

    # deviation from straight line
    deviations = []
    for i in range(len(xs)):
        num = abs((end_y - start_y)*xs[i] - (end_x - start_x)*ys[i] + end_x*start_y - end_y*start_x)
        den = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2 + 1e-6)
        deviations.append(num / den)

    mean_dev = np.mean(deviations)

    # ---------------- MULTI-TOUCH FEATURES ----------------
    # default values (single touch)
    start_ifd = 0
    end_ifd = 0
    scale_factor = 1

    if "fingers" in gesture[0] and len(gesture[0]["fingers"]) == 2:
        f_start = gesture[0]["fingers"]
        f_end = gesture[-1]["fingers"]

        start_ifd = np.linalg.norm(np.array(f_start[0]) - np.array(f_start[1]))
        end_ifd = np.linalg.norm(np.array(f_end[0]) - np.array(f_end[1]))

        scale_factor = end_ifd / (start_ifd + 1e-6)

    return [
        duration, start_x, start_y, end_x, end_y,
        mean_pressure, pressure_std,
        displacement,
        mean_speed, speed_std,
        slope, mean_dev,
        start_ifd, end_ifd, scale_factor
    ]


# ==========================================
# 2. GENERATE SAMPLE DATA (FOR TESTING)
# ==========================================

def generate_dummy_gesture(label):
    gesture = []
    t = 0

    for i in range(10):
        gesture.append({
            "time": t,
            "x": np.random.rand(),
            "y": np.random.rand(),
            "pressure": np.random.rand(),
        })
        t += np.random.rand()

    return gesture, label


def create_dataset(n=100):
    data = []
    labels = []

    for _ in range(n):
        g, label = generate_dummy_gesture("scroll")
        data.append(extract_features(g))
        labels.append(label)

    for _ in range(n):
        g, label = generate_dummy_gesture("swipe")
        data.append(extract_features(g))
        labels.append(label)

    for _ in range(n):
        g, label = generate_dummy_gesture("pinch")
        data.append(extract_features(g))
        labels.append(label)

    return np.array(data), np.array(labels)


# ==========================================
# 3. TRAIN SVM MODEL
# ==========================================

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True))
    ])

    param_grid = {
        "svm__kernel": ["rbf", "linear"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("Generating dataset...")
    X, y = create_dataset(100)

    print("Training SVM model...")
    model = train_model(X, y)

    # Save model
    joblib.dump(model, "gesture_svm_model.pkl")
    print("\nModel saved as gesture_svm_model.pkl")