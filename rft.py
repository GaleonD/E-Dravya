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

def check_concentration_thresholds(df, label_col="Potency", conc_col="Concentration"):
    if conc_col not in df.columns:
        return {}
    mapping = {}
    for lv in sorted(df[label_col].unique()):
        vals = df[df[label_col] == lv][conc_col]
        if len(vals) == 0:
            mapping[lv] = None
        else:
            mapping[lv] = float(np.median(vals))
    thresholds = {}
    sorted_items = sorted(mapping.items(), key=lambda x: (x[1] if x[1] is not None else -1))
    mids = []
    nums = [v for k,v in sorted_items if v is not None]
    for i in range(len(nums)-1):
        mids.append((nums[i] + nums[i+1]) / 2.0)
    def rule_predict(c):
        if np.isnan(c) or c is None:
            return None
        best = None
        bestd = None
        for k, med in mapping.items():
            if med is None:
                continue
            d = abs(c - med)
            if bestd is None or d < bestd:
                bestd = d
                best = k
        return best
    tmp = df.dropna(subset=[conc_col, label_col])
    if len(tmp) == 0:
        acc = None
        cm = None
    else:
        preds = tmp[conc_col].apply(rule_predict).astype(str)
        le_local = LabelEncoder().fit(tmp[label_col].astype(str))
        try:
            y_true = le_local.transform(tmp[label_col].astype(str))
            y_pred = le_local.transform(preds)
            acc = float((y_true == y_pred).mean())
            cm = confusion_matrix(y_true, y_pred)
        except Exception:
            acc = None
            cm = None
    return {"mapping": mapping, "thresholds": mids, "accuracy": acc, "confusion_matrix": cm, "label_order": sorted(df[label_col].unique())}

def train():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    if "Potency" not in df.columns:
        raise RuntimeError("Expected label column 'Potency' in CSV")

    fit_preprocessor(df, ARTIFACT_DIR, do_scale=False)
    artifacts = load_artifacts(ARTIFACT_DIR)

    X = transform_df(df[artifacts["feature_columns"]], artifacts)
    y_raw = df["Potency"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
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

    if "Concentration" in df.columns:
        res = check_concentration_thresholds(df)
        print("\nConcentration-threshold rule analysis:")
        print(" Mapping (concentration -> majority potency):", res.get("mapping"))
        print(" Threshold midpoints:", res.get("thresholds"))
        print(f" Rule Accuracy on dataset: {res.get('accuracy')}")
        print(" Rule confusion matrix (rows=true, cols=pred) with label order:", res.get("label_order"))
        print(res.get("confusion_matrix"))

if __name__ == "__main__":
    train()

