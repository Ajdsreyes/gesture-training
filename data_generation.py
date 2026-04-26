import numpy as np
from feature_extraction import extract_features

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