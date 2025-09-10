#only required for EDGE INFERENCE (Raspberry Pi add-on, etc.)
#scope for future expansion (?), in scenarios where cloud connectivity is limited or local processing is preferred.
#NOT NEEDED RIGHT NOW FOR BASIC FUNCTIONALITY

import os
import sys
import json
import argparse
import time
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "rft_herb_model.pkl")
LE_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")
META_PATH = os.path.join(ARTIFACT_DIR, "preprocess_metadata.pkl")

try:
    from preprocess import preprocess_input
except Exception:
    preprocess_input = None

def load_artifacts(model_path=MODEL_PATH, le_path=LE_PATH, meta_path=META_PATH):
    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None
    metadata = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    feature_columns = metadata.get("feature_columns", metadata.get("feature_cols", None))
    return model, label_encoder, feature_columns

def dict_to_vector(raw, feature_columns):
    if feature_columns is None:
        keys = sorted([k for k,v in raw.items() if isinstance(v,(int,float))])
        feature_columns = keys
    vec = [float(raw.get(c, 0.0)) for c in feature_columns]
    return np.asarray(vec, dtype=float).reshape(1, -1)

def predict_from_raw(model, label_encoder, feature_columns, raw):
    if preprocess_input is not None:
        try:
            vec = preprocess_input(raw, feature_columns=feature_columns)
        except Exception:
            vec = dict_to_vector(raw, feature_columns)
    else:
        vec = dict_to_vector(raw, feature_columns)
    pred = model.predict(vec)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(vec).max(axis=1)[0]
        except Exception:
            proba = None
    else:
        proba = None
    if label_encoder is not None:
        try:
            pred_label = label_encoder.inverse_transform(pred)[0]
        except Exception:
            pred_label = str(pred[0])
    else:
        pred_label = str(pred[0])
    out = {"prediction": pred_label, "confidence": float(proba) if proba is not None else None, "timestamp": int(time.time())}
    return out

def predict_json_file(model, label_encoder, feature_columns, json_path):
    with open(json_path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        out = predict_from_raw(model, label_encoder, feature_columns, obj)
        print(json.dumps(out, indent=2))
    elif isinstance(obj, list):
        results = []
        for raw in obj:
            results.append(predict_from_raw(model, label_encoder, feature_columns, raw))
        print(json.dumps(results, indent=2))
    else:
        raise ValueError("Unsupported JSON structure for prediction")

def predict_csv(model, label_encoder, feature_columns, csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        raw = r.to_dict()
        rows.append(predict_from_raw(model, label_encoder, feature_columns, raw))
    print(json.dumps(rows, indent=2))

def start_mqtt_loop(model, label_encoder, feature_columns, broker, port, topic):
    import paho.mqtt.client as mqtt
    def on_connect(client, userdata, flags, rc):
        client.subscribe(topic)
    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
            res = predict_from_raw(model, label_encoder, feature_columns, data)
            print(json.dumps(res))
        except Exception as e:
            print("Error:", e, file=sys.stderr)
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, 60)
    client.loop_forever()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--csv")
    parser.add_argument("--mqtt", action="store_true")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="herb/sensor/data")
    parser.add_argument("--model")
    parser.add_argument("--le")
    parser.add_argument("--meta")
    args = parser.parse_args()
    model_path = args.model if args.model else MODEL_PATH
    le_path = args.le if args.le else LE_PATH
    meta_path = args.meta if args.meta else META_PATH
    model, label_encoder, feature_columns = load_artifacts(model_path, le_path, meta_path)
    if args.json:
        predict_json_file(model, label_encoder, feature_columns, args.json)
    elif args.csv:
        predict_csv(model, label_encoder, feature_columns, args.csv)
    elif args.mqtt:
        start_mqtt_loop(model, label_encoder, feature_columns, args.broker, args.port, args.topic)
    else:
        print("Provide --json or --csv or --mqtt")
        sys.exit(1)

if __name__ == "__main__":
    main()
