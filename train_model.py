import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_CSV = "gesture_data.csv"
OUT_MODEL = "gesture_model.joblib"

def main():
    df = pd.read_csv(DATA_CSV)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Good baseline model for landmark classification
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, OUT_MODEL)
    print(f"Saved model to: {OUT_MODEL}")

if __name__ == "__main__":
    main()