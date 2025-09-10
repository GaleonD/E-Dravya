#the big boy, trains RandomForest on preprocessed data and saves model and label encoder

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preproc import fit_preprocessor, load_artifacts, transform_df

CSV_PATH = os.path.join(os.path.dirname(__file__), "main_dataset.csv")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.2

def train():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    if "Herb_Name" not in df.columns:
        raise RuntimeError("Expected label column 'Herb_Name' in CSV")

    fit_preprocessor(df, ARTIFACT_DIR, do_scale=False)
    artifacts = load_artifacts(ARTIFACT_DIR)

    X = transform_df(df[artifacts["feature_columns"]], artifacts)
    y_raw = df["Herb_Name"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print("Train accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    feat_names = artifacts["feature_columns"]
    importances = clf.feature_importances_
    pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    print("Feature importances:")
    for n, imp in pairs:
        print(f"  {n:15s}: {imp:.4f}")

    model_path = os.path.join(ARTIFACT_DIR, "rft_herb_model.pkl")
    le_path = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")
    joblib.dump(clf, model_path)
    joblib.dump(le, le_path)
    print(f"\nSaved: {model_path}")
    print(f"Saved: {le_path}")

if __name__ == "__main__":
    train()

