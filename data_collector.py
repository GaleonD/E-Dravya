#basically regenerates CSV(s) from DynamoDB, with optional label decoding and preprocessing
#NOT NEEDED RIGHT NOW FOR BASIC FUNCTIONALITY

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import botocore
import pandas as pd

try:
    from preproc import load_artifacts, transform_raw
except Exception:
    load_artifacts = None
    transform_raw = None

import pickle

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--start", help="start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="end date YYYY-MM-DD (inclusive)")
    p.add_argument("--limit", type=int, default=0, help="max items to fetch (0 = all)")
    p.add_argument("--sample-rate", type=float, default=1.0, help="sample rate 0-1 for downsampling")
    p.add_argument("--label-encoder", help="path to label_encoder.pkl to map numeric labels")
    p.add_argument("--preproc-artifacts", help="path to preproc artifacts for transform_raw")
    p.add_argument("--train-csv", default="train_ready.csv", help="filename for model-ready CSV (optional)")
    p.add_argument("--raw-csv", default="raw_scans.csv", help="filename for raw flattened CSV")
    p.add_argument("--s3-bucket", help="optional S3 bucket to upload outputs")
    p.add_argument("--s3-prefix", default="", help="optional S3 prefix (folder) for uploads")
    p.add_argument("--region", default=None, help="AWS region (overrides env)")
    return p.parse_args()

def to_timestamp_bounds(start: Optional[str], end: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    import datetime
    ts_start = None
    ts_end = None
    if start:
        dt = datetime.datetime.fromisoformat(start)
        ts_start = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    if end:
        dt = datetime.datetime.fromisoformat(end)
        dt = dt.replace(hour=23, minute=59, second=59)
        ts_end = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    return ts_start, ts_end

def dynamo_scan(table_name: str, region: Optional[str], start_ts: Optional[int], end_ts: Optional[int],
                limit: int = 0) -> Iterable[Dict[str, Any]]:
    sess = boto3.session.Session()
    if region:
        ddb = sess.client("dynamodb", region_name=region)
    else:
        ddb = sess.client("dynamodb")
    paginator = ddb.get_paginator("scan")
    filter_expression = None
    expression_attr = {}
    if start_ts is not None and end_ts is not None:
        filter_expression = "timestamp BETWEEN :s AND :e"
        expression_attr[":s"] = {"N": str(start_ts)}
        expression_attr[":e"] = {"N": str(end_ts)}
    elif start_ts is not None:
        filter_expression = "timestamp >= :s"
        expression_attr[":s"] = {"N": str(start_ts)}
    elif end_ts is not None:
        filter_expression = "timestamp <= :e"
        expression_attr[":e"] = {"N": str(end_ts)}

    scan_kwargs = {"TableName": table_name}
    if filter_expression:
        scan_kwargs.update({"FilterExpression": filter_expression,
                            "ExpressionAttributeValues": expression_attr})

    fetched = 0
    try:
        for page in paginator.paginate(**scan_kwargs):
            items = page.get("Items", [])
            for it in items:
                # convert DynamoDB JSON to plain python types
                plain = _dynamo_item_to_plain(it)
                yield plain
                fetched += 1
                if limit and fetched >= limit:
                    return
    except botocore.exceptions.ClientError as e:
        raise RuntimeError(f"DynamoDB scan failed: {e}")

def _dynamo_item_to_plain(item: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in item.items():
        if "S" in v:
            out[k] = v["S"]
        elif "N" in v:
            n = v["N"]
            if "." in n:
                out[k] = float(n)
            else:
                out[k] = int(n)
        elif "BOOL" in v:
            out[k] = bool(v["BOOL"])
        elif "M" in v:
            out[k] = _dynamo_map_to_plain(v["M"])
        elif "L" in v:
            out[k] = [_dynamo_value_to_plain(x) for x in v["L"]]
        else:
            out[k] = None
    return out

def _dynamo_map_to_plain(m: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for kk, vv in m.items():
        out[kk] = _dynamo_value_to_plain(vv)
    return out

def _dynamo_value_to_plain(v: Dict[str, Any]) -> Any:
    if "S" in v:
        return v["S"]
    if "N" in v:
        n = v["N"]
        return float(n) if "." in n else int(n)
    if "BOOL" in v:
        return bool(v["BOOL"])
    if "M" in v:
        return _dynamo_map_to_plain(v["M"])
    if "L" in v:
        return [_dynamo_value_to_plain(x) for x in v["L"]]
    return None

def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    out["device_id"] = rec.get("device_id")
    out["timestamp"] = rec.get("timestamp")
    sensors = rec.get("sensor_readings") or {}
    if isinstance(sensors, str):
        try:
            sensors = json.loads(sensors)
        except Exception:
            sensors = {}
    for k, v in sensors.items():
        out[k] = v
    out["prediction"] = rec.get("prediction")
    out["confidence"] = rec.get("confidence")
    out["adulteration_alert"] = rec.get("adulteration_alert")
    out["model_version"] = rec.get("model_version")
    out["source"] = rec.get("source")
    out["raw_json"] = json.dumps(rec)
    return out

def load_label_encoder(path: Optional[str]):
    if not path:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def upload_to_s3(local_path: str, bucket: str, key: str, region: Optional[str]):
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    s3.upload_file(local_path, bucket, key)

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = to_timestamp_bounds(args.start, args.end)
    label_enc = load_label_encoder(args.label_encoder) if args.label_encoder else None
    preproc_artifacts = None
    if args.preproc_artifacts:
        if load_artifacts is None:
            print("preproc.load_artifacts not available; cannot apply preproc transform.", file=sys.stderr)
            args.preproc_artifacts = None
        else:
            preproc_artifacts = load_artifacts(args.preproc_artifacts)

    seen = set()
    rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    count = 0
    for rec in dynamo_scan(args.table, args.region, start_ts, end_ts, limit=args.limit):
        # dedupe
        dev = rec.get("device_id")
        ts = rec.get("timestamp")
        if dev is None or ts is None:
            continue
        key = f"{dev}#{ts}"
        if key in seen:
            continue
        seen.add(key)

        # sampling
        import random
        if args.sample_rate < 1.0 and random.random() > args.sample_rate:
            continue

        flat = flatten_record(rec)
        rows.append(flat)

        if label_enc and flat.get("prediction") is not None:
            try:
                if isinstance(flat["prediction"], (int, float)):
                    lbl = label_enc.inverse_transform([int(flat["prediction"])])[0]
                    flat["prediction_label"] = lbl
                else:
                    flat["prediction_label"] = str(flat["prediction"])
            except Exception:
                flat["prediction_label"] = str(flat.get("prediction"))
        else:
            flat["prediction_label"] = flat.get("prediction")

        if preproc_artifacts:
            try:
                features = transform_raw(flat.get("sensor_readings") or json.loads(flat.get("raw_json") or "{}"), preproc_artifacts)
                feat_list = list(map(float, features.reshape(-1).tolist()))
                for i, col in enumerate(preproc_artifacts["feature_columns"]):
                    flat[col] = feat_list[i]
                train_rows.append(flat)
            except Exception:
                pass

        count += 1
        if args.limit and count >= args.limit:
            break

    # write raw CSV
    raw_path = out_dir / args.raw_csv
    tmpf = out_dir / (args.raw_csv + ".tmp")
    if rows:
        # get union of keys for header
        header = sorted({k for r in rows for k in r.keys()})
        with open(tmpf, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        tmpf.replace(raw_path)

    # write train-ready CSV if requested
    train_path = out_dir / args.train_csv
    if train_rows:
        header = sorted({k for r in train_rows for k in r.keys()})
        tmpf2 = out_dir / (args.train_csv + ".tmp")
        with open(tmpf2, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in train_rows:
                w.writerow(r)
        tmpf2.replace(train_path)

    # optional S3 upload
    if args.s3_bucket:
        if raw_path.exists():
            key = os.path.join(args.s3_prefix, raw_path.name) if args.s3_prefix else raw_path.name
            upload_to_s3(str(raw_path), args.s3_bucket, key, args.region)
        if train_rows and train_path.exists():
            key = os.path.join(args.s3_prefix, train_path.name) if args.s3_prefix else train_path.name
            upload_to_s3(str(train_path), args.s3_bucket, key, args.region)

    print(f"Done. Raw rows: {len(rows)}. Train-ready rows: {len(train_rows)}. Outputs written to {out_dir}")

if __name__ == "__main__":
    main()
