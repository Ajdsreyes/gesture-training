from data_generation import create_dataset
from model_training import train_model
import joblib

if __name__ == "__main__":

    print("Generating dataset...")
    X, y = create_dataset(100)

    print("Training SVM model...")
    model = train_model(X, y)

    joblib.dump(model, "gesture_svm_model.pkl")
    print("\nModel saved as gesture_svm_model.pkl")