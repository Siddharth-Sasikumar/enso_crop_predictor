import json
import joblib
import requests
import warnings
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "final_crop_dataset_complete.csv"
MODEL_PATH   = BASE_DIR / "python" / "rainfall_model.pkl"
ENCODER_PATH = BASE_DIR / "python" / "encoders.pkl"
FORECAST_PATH = BASE_DIR / "forecast.json"


# ============================================
# COLUMN NORMALISATION
# ============================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = {}
    for c in df.columns:
        key = c.strip()
        low = key.lower().strip().replace(" ", "_")

        if low in {"state", "state_name", "state_name"}:
            cleaned[key] = "State"
        elif low in {"season", "season_name"}:
            cleaned[key] = "Season"
        elif low in {"year", "crop_year", "cropyear"}:
            cleaned[key] = "Year"
        elif low in {"enso_phase", "enso", "enso_phase"}:
            cleaned[key] = "ENSO_Phase"
        elif low in {"avg_oni", "oni", "oni_value", "average_oni"}:
            cleaned[key] = "Avg_ONI"
        elif low in {"annual_rainfall", "rainfall", "annual_rainfall", "rainfall_mm"}:
            cleaned[key] = "Annual_Rainfall"

    return df.rename(columns=cleaned)


# ============================================
# ONI FETCH
# ============================================

def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def fetch_live_oni():
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        lines = [x.strip() for x in r.text.splitlines() if x.strip()]
        vals = []
        for line in lines[1:]:
            parts = line.split()
            nums = [p for p in parts if _is_float(p)]
            if nums:
                vals.append(float(nums[-1]))
        return vals[-1] if vals else None
    except Exception:
        return None


def oni_to_phase(oni):
    if oni is None:
        return "Neutral"
    if oni >= 0.5:
        return "El Nino"
    if oni <= -0.5:
        return "La Nina"
    return "Neutral"


def intensity_from_oni(oni):
    if oni is None:
        return "Neutral"
    a = abs(float(oni))
    if a >= 1.5:
        return "Strong"
    elif a >= 1.0:
        return "Moderate"
    elif a >= 0.5:
        return "Weak"
    return "Neutral"


# ============================================
# DATASET
# ============================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)

    required = ["State", "Season", "Year", "ENSO_Phase", "Annual_Rainfall"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df.dropna(subset=required).copy()
    df["Year"]             = pd.to_numeric(df["Year"],             errors="coerce")
    df["Annual_Rainfall"]  = pd.to_numeric(df["Annual_Rainfall"],  errors="coerce")

    if "Avg_ONI" in df.columns:
        df["Avg_ONI"] = pd.to_numeric(df["Avg_ONI"], errors="coerce")

    df = df.dropna(subset=["Year", "Annual_Rainfall"])
    df["Year"] = df["Year"].astype(int)

    return df


# ============================================
# FEATURE ENGINEERING
# FIX: Added lag features and rolling stats
# ============================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag and rolling features that significantly
    improve RandomForest rainfall prediction accuracy.
    """
    df = df.sort_values(["State", "Season", "Year"]).copy()

    grp = df.groupby(["State", "Season"])["Annual_Rainfall"]

    # Lag-1: previous year's rainfall for same state+season
    df["Rainfall_Lag1"] = grp.shift(1)

    # Lag-2: two years ago
    df["Rainfall_Lag2"] = grp.shift(2)

    # 3-year rolling mean (excluding current year)
    df["Rainfall_Roll3"] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # ONI lag-1 if available
    if "Avg_ONI" in df.columns:
        df["ONI_Lag1"] = df.groupby(["State", "Season"])["Avg_ONI"].shift(1)
    else:
        df["ONI_Lag1"] = 0.0

    # Fill NaN lag values with state+season climatology
    for col in ["Rainfall_Lag1", "Rainfall_Lag2", "Rainfall_Roll3"]:
        df[col] = df[col].fillna(
            df.groupby(["State", "Season"])["Annual_Rainfall"].transform("mean")
        )

    df["ONI_Lag1"] = df["ONI_Lag1"].fillna(0.0)

    return df


# ============================================
# MODEL TRAINING
# FIX: More features, GradientBoosting ensemble
# ============================================

def train_model():
    df = load_dataset()
    df = build_features(df)

    le_state  = LabelEncoder()
    le_season = LabelEncoder()
    le_enso   = LabelEncoder()

    df["state_enc"]  = le_state.fit_transform(df["State"])
    df["season_enc"] = le_season.fit_transform(df["Season"])
    df["enso_enc"]   = le_enso.fit_transform(df["ENSO_Phase"])

    feature_cols = [
        "state_enc", "season_enc", "enso_enc", "Year",
        "Rainfall_Lag1", "Rainfall_Lag2", "Rainfall_Roll3", "ONI_Lag1"
    ]

    train_df = df.dropna(subset=feature_cols)
    X = train_df[feature_cols]
    y = train_df["Annual_Rainfall"]

    # FIX: Use GradientBoosting for better accuracy on tabular data
    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(
        {
            "state": le_state,
            "season": le_season,
            "enso": le_enso,
            "feature_cols": feature_cols
        },
        ENCODER_PATH
    )

    return model, le_state, le_season, le_enso, feature_cols


def load_or_train():
    if MODEL_PATH.exists() and ENCODER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        enc   = joblib.load(ENCODER_PATH)
        feature_cols = enc.get(
            "feature_cols",
            ["state_enc", "season_enc", "enso_enc", "Year",
             "Rainfall_Lag1", "Rainfall_Lag2", "Rainfall_Roll3", "ONI_Lag1"]
        )
        return model, enc["state"], enc["season"], enc["enso"], feature_cols
    return train_model()


# ============================================
# UNCERTAINTY ESTIMATE
# FIX: Bootstrap residual estimate for GBR
# (GBR has no tree ensemble like RF)
# ============================================

def estimate_uncertainty_gbr(model, x_row, df, state, season):
    """
    For GradientBoosting we use historical residual spread
    as a proxy for confidence interval.
    """
    sub = df[(df["State"] == state) & (df["Season"] == season)]["Annual_Rainfall"]
    if sub.shape[0] < 3:
        pred = float(model.predict(x_row)[0])
        return round(pred * 0.85, 1), round(pred * 1.15, 1)

    std = float(sub.std())
    pred = float(model.predict(x_row)[0])
    return round(max(0, pred - 1.28 * std), 1), round(pred + 1.28 * std, 1)


# ============================================
# HISTORICAL LOOKUPS
# ============================================

def get_historical_oni_phase(df, year):
    yr_df = df[df["Year"] == int(year)].copy()

    if yr_df.empty:
        return 0.0, "Neutral"

    if "Avg_ONI" in yr_df.columns and yr_df["Avg_ONI"].dropna().shape[0] > 0:
        oni = float(yr_df["Avg_ONI"].dropna().mean())
        return round(oni, 2), oni_to_phase(oni)

    mode_phase = yr_df["ENSO_Phase"].dropna().astype(str).mode()
    phase = mode_phase.iloc[0] if not mode_phase.empty else "Neutral"
    phase_to_oni = {"El Nino": 0.8, "La Nina": -0.8, "Neutral": 0.0}
    oni = phase_to_oni.get(phase, 0.0)
    return oni, phase


def get_historical_rainfall(df, state, season, year):
    sub = df[
        (df["State"] == state) &
        (df["Season"] == season) &
        (df["Year"] == int(year))
    ]

    if not sub.empty:
        return float(sub["Annual_Rainfall"].mean()), "observed"

    fallback = df[(df["State"] == state) & (df["Season"] == season)]
    if not fallback.empty:
        return float(fallback["Annual_Rainfall"].mean()), "climatology"

    return 500.0, "default"


# ============================================
# LAG VALUES FOR FUTURE PREDICTION
# ============================================

def get_lag_values(df, state, season, year):
    """
    Pull the most recent available lag features
    for a future (unseen) year.
    """
    hist = df[
        (df["State"] == state) & (df["Season"] == season)
    ].sort_values("Year")

    if hist.empty:
        return 500.0, 500.0, 500.0, 0.0

    available_years = hist["Year"].values
    recent = hist.tail(3)

    lag1 = float(hist[hist["Year"] == year - 1]["Annual_Rainfall"].mean()) \
           if (year - 1) in available_years else float(recent["Annual_Rainfall"].iloc[-1])

    lag2 = float(hist[hist["Year"] == year - 2]["Annual_Rainfall"].mean()) \
           if (year - 2) in available_years else float(recent["Annual_Rainfall"].mean())

    roll3 = float(recent["Annual_Rainfall"].mean())

    oni_lag1 = 0.0
    if "Avg_ONI" in hist.columns:
        oni_row = hist[hist["Year"] == year - 1]["Avg_ONI"].dropna()
        if not oni_row.empty:
            oni_lag1 = float(oni_row.mean())

    return lag1, lag2, roll3, oni_lag1


# ============================================
# FORECAST JSON
# ============================================

def write_forecast_json(pred_result, output_file=FORECAST_PATH):
    forecast = {
        "enso_phase":     pred_result["enso_phase"],
        "enso_intensity": intensity_from_oni(pred_result["oni_value"]),
        "rainfall_input": pred_result["predicted_rainfall_mm"]
    }
    with open(output_file, "w") as f:
        json.dump(forecast, f, indent=2)


# ============================================
# MAIN PREDICTION FUNCTION
# ============================================

def predict_climate(state: str, season: str, year: int,
                    enso_mode="historical", manual_phase=None):

    model, le_state, le_season, le_enso, feature_cols = load_or_train()
    df = load_dataset()
    df = build_features(df)

    # FIX: Input validation
    if state not in le_state.classes_:
        raise ValueError(f"Unknown state: '{state}'. "
                         f"Valid states: {list(le_state.classes_)[:5]} ...")

    if season not in le_season.classes_:
        raise ValueError(f"Unknown season: '{season}'. "
                         f"Valid seasons: {list(le_season.classes_)}")

    # Determine ENSO
    if enso_mode == "live":
        oni   = fetch_live_oni()
        phase = oni_to_phase(oni)
    elif enso_mode == "manual":
        phase   = manual_phase if manual_phase else "Neutral"
        oni_map = {"El Nino": 0.8, "Neutral": 0.0, "La Nina": -0.8}
        oni     = oni_map.get(phase, 0.0)
    else:
        oni, phase = get_historical_oni_phase(df, year)

    if phase not in le_enso.classes_:
        phase = "Neutral"

    # Historical path (observed data exists)
    if enso_mode == "historical" and int(year) <= int(df["Year"].max()):
        rainfall, prediction_source = get_historical_rainfall(df, state, season, year)
        ci_low = ci_high = round(rainfall, 1)

    # Future / ML path
    else:
        lag1, lag2, roll3, oni_lag1 = get_lag_values(df, state, season, year)

        x_row = pd.DataFrame({
            "state_enc":      [le_state.transform([state])[0]],
            "season_enc":     [le_season.transform([season])[0]],
            "enso_enc":       [le_enso.transform([phase])[0]],
            "Year":           [int(year)],
            "Rainfall_Lag1":  [lag1],
            "Rainfall_Lag2":  [lag2],
            "Rainfall_Roll3": [roll3],
            "ONI_Lag1":       [oni_lag1]
        })

        rainfall = float(model.predict(x_row)[0])
        ci_low, ci_high = estimate_uncertainty_gbr(model, x_row, df, state, season)
        prediction_source = "ml_model"

    # Anomaly vs historical normal
    hist   = df[(df["State"] == state) & (df["Season"] == season)]
    normal = float(hist["Annual_Rainfall"].mean()) if not hist.empty else rainfall
    anomaly_pct = ((rainfall - normal) / normal * 100) if normal != 0 else 0.0

    if anomaly_pct > 10:
        category = "Above Normal"
    elif anomaly_pct < -10:
        category = "Below Normal"
    else:
        category = "Normal"

    result = {
        "state":                  state,
        "season":                 season,
        "year":                   int(year),
        "enso_mode":              enso_mode,
        "enso_phase":             phase,
        "oni_value":              None if oni is None else round(float(oni), 2),
        "enso_intensity":         intensity_from_oni(oni),
        "predicted_rainfall_mm":  round(rainfall, 1),
        "historical_normal_mm":   round(normal, 1),
        "anomaly_pct":            round(anomaly_pct, 1),
        "rainfall_category":      category,
        "confidence_interval_mm": [ci_low, ci_high],
        "prediction_source":      prediction_source
    }

    write_forecast_json(result)
    return result


# ============================================
# CLI ENTRY POINT
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state",        required=True)
    parser.add_argument("--season",       required=True)
    parser.add_argument("--year",         required=True, type=int)
    parser.add_argument("--enso-mode",
                        choices=["historical", "live", "manual"],
                        default="historical")
    parser.add_argument("--manual-phase",
                        choices=["El Nino", "Neutral", "La Nina"],
                        default=None)
    args = parser.parse_args()

    result = predict_climate(
        args.state,
        args.season,
        args.year,
        enso_mode=args.enso_mode,
        manual_phase=args.manual_phase
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()