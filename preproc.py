import json
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: List[str] = [
    "pH",
    "TDS_ppm",
    "ORP_mV",
    "Temperature_C",
    "Color_R",
    "Color_G",
    "Color_B",
]

KEY_ALIASES = {
    "r": "Color_R",
    "g": "Color_G",
    "b": "Color_B",
    "orp": "ORP_mV",
    "TDS": "TDS_ppm",
    "tds": "TDS_ppm",
    "temp": "Temperature_C",
    "temperature": "Temperature_C",
    "pH": "pH",
    "ph": "pH",
}

def _map_keys(raw: Dict) -> Dict:
    out = {}
    for k, v in raw.items():
        if k in FEATURE_COLUMNS:
            out[k] = v
        elif k in KEY_ALIASES:
            out[KEY_ALIASES[k]] = v
        else:
            out[k] = v
    return out

def apply_calibration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Color_R" in df.columns:
        df["Color_R"] = df["Color_R"] * 1.0
    if "Color_G" in df.columns:
        df["Color_G"] = df["Color_G"] * 1.0
    if "Color_B" in df.columns:
        df["Color_B"] = df["Color_B"] * 1.0
    if "ORP_mV" in df.columns:
        df["ORP_mV"] = df["ORP_mV"] + 0.0
    if "pH" in df.columns:
        df["pH"] = df["pH"] + 0.0
    if "TDS_ppm" in df.columns:
        df["TDS_ppm"] = df["TDS_ppm"] * 1.0
    if "Temperature_C" in df.columns:
        df["Temperature_C"] = df["Temperature_C"] * 1.0
    return df

def fit_preprocessor(df: pd.DataFrame, out_dir: str or Path, do_scale: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    df_feat = df.reindex(columns=present).copy()
    df_feat = apply_calibration(df_feat)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_feat.values)
    imputed = imputer.transform(df_feat.values)
    scaler = None
    if do_scale:
        scaler = StandardScaler()
        scaler.fit(imputed)
        transformed = scaler.transform(imputed)
    else:
        transformed = imputed
    with open(out_dir / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    if scaler is not None:
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(present, f)
    return {"imputer": imputer, "scaler": scaler, "feature_columns": present}

def load_artifacts(path: str or Path):
    p = Path(path)
    with open(p / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    with open(p / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    scaler = None
    if (p / "scaler.pkl").exists():
        with open(p / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    return {"imputer": imputer, "scaler": scaler, "feature_columns": feature_columns}

def transform_df(df: pd.DataFrame, artifacts: Dict):
    feature_columns = artifacts["feature_columns"]
    df_sel = df.reindex(columns=feature_columns)
    df_sel = apply_calibration(df_sel)
    imputed = artifacts["imputer"].transform(df_sel.values)
    scaler = artifacts.get("scaler")
    if scaler is not None:
        transformed = scaler.transform(imputed)
    else:
        transformed = imputed
    return transformed

def transform_raw(raw: Dict, artifacts: Dict):
    mapped = _map_keys(raw)
    df = pd.DataFrame([mapped])
    arr = transform_df(df, artifacts)
    return arr
