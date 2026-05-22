import json
from feature_extraction import extract_features

with open("gesture.json", "r") as f:
    data = json.load(f)

gesture = data["touch_position"]

features = extract_features(gesture)

print(features)