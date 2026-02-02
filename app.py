# app.py
# Streamlit: Energy Efficiency Model Lab (2025)
# ------------------------------------------------------------
# Features:
# - Load CSV (or auto-generate a 500k-row synthetic dataset like UCI Energy)
# - Regression (HeatingLoad / CoolingLoad) or Classification (binned HeatingLoad)
# - Model selection (skips gracefully if libraries are missing)
# - Timed training/inference + CodeCarbon emissions (if available)
# - Subsampling for speed, CV option, residual/confusion plots
# - Scenario "what-if" prediction UI

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

# Optional models: import safely
def safe_import(import_str, obj_name):
    try:
        module = __import__(import_str, fromlist=[obj_name])
        return getattr(module, obj_name)
    except Exception:
        return None

XGBRegressor = safe_import("xgboost", "XGBRegressor")
XGBClassifier = safe_import("xgboost", "XGBClassifier")
LGBMRegressor = safe_import("lightgbm", "LGBMRegressor")
LGBMClassifier = safe_import("lightgbm", "LGBMClassifier")
CatBoostRegressor = safe_import("catboost", "CatBoostRegressor")
CatBoostClassifier = safe_import("catboost", "CatBoostClassifier")

# CodeCarbon tracker (optional)
def get_emissions_tracker():
    try:
        from codecarbon import EmissionsTracker
        return EmissionsTracker(project_name="EnergyEfficiencyApp", log_level="error")
    except Exception:
        return None

st.set_page_config(page_title="Energy Efficiency Model Lab", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def generate_synthetic_500k(seed=42, n_rows=500_000):
    rng = np.random.default_rng(seed)

    relative_compactness = np.round(rng.uniform(0.5, 1.0, n_rows), 3)
    surface_area = np.round(rng.uniform(50, 500, n_rows), 2)
    wall_area = np.round(rng.uniform(20, 300, n_rows), 2)
    roof_area = np.round(rng.uniform(30, 200, n_rows), 2)
    overall_height = np.round(rng.uniform(2.5, 10, n_rows), 2)

    orientation = rng.choice(['North', 'South', 'East', 'West'], n_rows)
    glazing_area_dist = rng.choice(
        ['Uniform', 'North-heavy', 'South-heavy', 'East-heavy', 'West-heavy'], n_rows
    )
    building_type = rng.choice(['Residential', 'Commercial', 'Industrial'], n_rows)

    # Subtle relationships to mimic realism (2025-ish constructions)
    heating_load = np.round(
        8
        + 35 * (1.05 - relative_compactness)                # worse compactness => more heat loss
        + 0.015 * (surface_area - 200)
        + 0.01 * (wall_area - 120)
        + 0.02 * (roof_area - 90)
        + 0.5 * (overall_height - 3)
        + rng.normal(0, 4, n_rows),
        2
    )

    cooling_load = np.round(
        10
        + 28 * (relative_compactness - 0.7)                 # more compact => harder to cool if gains present
        + 0.012 * (surface_area - 200)
        + 0.008 * (wall_area - 120)
        + 0.015 * (roof_area - 90)
        + 0.3 * (overall_height - 3)
        + np.select(
            [glazing_area_dist == 'South-heavy', glazing_area_dist == 'West-heavy'],
            [3.0, 1.5], 0.0
        )
        + rng.normal(0, 4, n_rows),
        2
    )

    df = pd.DataFrame({
        'RelativeCompactness': relative_compactness,
        'SurfaceArea': surface_area,
        'WallArea': wall_area,
        'RoofArea': roof_area,
        'OverallHeight': overall_height,
        'Orientation': orientation,
        'GlazingAreaDistribution': glazing_area_dist,
        'BuildingType': building_type,
        'HeatingLoad': heating_load,
        'CoolingLoad': cooling_load
    })

    return df

def build_model_zoo(task):
    models = {}
    if task == "Regression":
        models["LinearRegression"] = (LinearRegression(), True)
        models["Ridge"] = (Ridge(), True)
        models["Lasso"] = (Lasso(), True)
        models["RandomForestRegressor"] = (RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42), True)
        models["GradientBoostingRegressor"] = (GradientBoostingRegressor(random_state=42), True)
        if XGBRegressor:
            models["XGBRegressor"] = (XGBRegressor(
                n_estimators=400, learning_rate=0.05, subsample=0.7,
                colsample_bytree=0.8, random_state=42, tree_method="hist"
            ), True)
        if LGBMRegressor:
            models["LGBMRegressor"] = (LGBMRegressor(random_state=42), True)
        if CatBoostRegressor:
            models["CatBoostRegressor"] = (CatBoostRegressor(verbose=0, random_state=42), True)
    else:
        models["LogisticRegression"] = (LogisticRegression(max_iter=2000, n_jobs=None), False)
        models["RandomForestClassifier"] = (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42), False)
        models["GradientBoostingClassifier"] = (GradientBoostingClassifier(random_state=42), False)
        if XGBClassifier:
            models["XGBClassifier"] = (XGBClassifier(
                n_estimators=400, learning_rate=0.05, subsample=0.7,
                colsample_bytree=0.8, random_state=42, tree_method="hist",
                eval_metric="mlogloss"
            ), False)
        if LGBMClassifier:
            models["LGBMClassifier"] = (LGBMClassifier(random_state=42), False)
        if CatBoostClassifier:
            models["CatBoostClassifier"] = (CatBoostClassifier(verbose=0, random_state=42), False)
    return models

def info(msg):
    st.info(msg, icon="ℹ️")

def success(msg):
    st.success(msg, icon="✅")

def warn(msg):
    st.warning(msg, icon="⚠️")

# ---------------------------
# Sidebar: Data & Settings
# ---------------------------
st.title("🏗️ Energy Efficiency Model Lab (Interactive)")

with st.sidebar:
    st.header("1) Data")
    src = st.radio(
        "Choose dataset source",
        ["Upload CSV", "Generate synthetic (500k rows)"],
        help="If you upload, ensure it has at least 9 columns similar to the UCI Energy schema."
    )
    uploaded = None
    df = None
    if src == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
    else:
        with st.spinner("Generating hyperrealistic 500k-row dataset..."):
            df = generate_synthetic_500k()

    if df is not None:
        st.write("Rows:", len(df), "| Columns:", len(df.columns))
        if st.checkbox("Preview head (first 10 rows)", value=False):
            st.dataframe(df.head(10), use_container_width=True)

    st.header("2) Task & Target")
    task = st.radio("Choose task", ["Regression", "Classification"], horizontal=True)
    default_target_reg = "HeatingLoad"
    numeric_cols = []
    cat_cols = []
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if task == "Regression":
        target = st.selectbox("Regression target", options=[c for c in numeric_cols if c in ["HeatingLoad", "CoolingLoad"]] or numeric_cols)
    else:
        # Classification: bin HeatingLoad by default; user can choose bins
        target = st.selectbox("Base column to bin into classes", options=[c for c in numeric_cols if c in ["HeatingLoad"]] or numeric_cols)
        n_bins = st.slider("Number of bins (classes)", 3, 6, 3)
        bin_strategy = st.radio("Binning strategy", ["quantile", "uniform"], index=0,
                                help="Quantile creates balanced classes; uniform uses equal-width ranges.")

    st.header("3) Split & Sample")
    test_size = st.slider("Test size (%)", 10) / 100.0
    random_state = st.number_input("Random seed", value=42, step=1)
    max_rows = st.number_input("Subsample rows for speed", min_value=0, value=200_000, step=50_000,
                               help="Training all 500k rows can be heavy. Subsample for rapid experiments.")

    st.header("4) Models & Options")
    # Build model zoo placeholder (need task)
    models_all = build_model_zoo(task)
    model_names = list(models_all.keys())
    selected_models = st.multiselect("Select models to compare", model_names, default=model_names[:4])

    scale_numeric = st.checkbox("Standardize numeric features (Linear/Ridge/Lasso/LogReg benefit)", value=True)
    use_cv = st.checkbox("5-fold cross-validation on train (time-consuming)", value=False)
    track_emissions = st.checkbox("Track energy with CodeCarbon (if installed)", value=True)

    st.header("5) Run")
    go = st.button("🚀 Train & Compare")

# ---------------------------
# Data checks & preprocessing
# ---------------------------
if df is None:
    info("Upload a CSV or generate a dataset from the sidebar to begin.")
    st.stop()

# Drop rows with na/inf just in case
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Enforce dedup (synthetic shouldn’t duplicate; but user CSV might)
df = df.drop_duplicates()

# Build features/target
features = [c for c in df.columns if c != target]
X = df[features].copy()

# Encode categoricals with one-hot
X = pd.get_dummies(X, drop_first=True)

# Build y
if task == "Regression":
    y = df[target].values
else:
    base = df[target].values
    # bin to classes
    if bin_strategy == "quantile":
        # qcut can fail if too many duplicates at edges; add tiny jitter
        jitter = (np.random.default_rng(int(random_state)).normal(0, 1e-6, size=base.shape))
        y = pd.qcut(base + jitter, q=n_bins, labels=False, duplicates="drop")
    else:
        y = pd.cut(base, bins=n_bins, labels=False, duplicates="drop")
    y = y.astype(int)

# Optional subsample
if max_rows and max_rows > 0 and len(X) > max_rows:
    X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y if task == "Classification" else None,
                                  random_state=int(random_state))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size,
    stratify=y if task == "Classification" else None,
    random_state=int(random_state)
)

# Scaling
scaler = None
if scale_numeric:
    num_cols = [c for c in X_train.columns if np.issubdtype(X_train[c].dtype, np.number)]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------
# Train / Evaluate
# ---------------------------
def evaluate_model(name, model, is_reg):
    tracker = get_emissions_tracker() if track_emissions else None
    emissions = None

    # Start emissions tracking
    if tracker is not None:
        try:
            tracker.start()
        except Exception:
            tracker = None  # disable silently if environment not supported

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Stop emissions tracking (training only)
    if tracker is not None:
        try:
            emissions = tracker.stop()  # in kg CO2e
        except Exception:
            emissions = None

    # Inference
    start_inf = time.time()
    preds = model.predict(X_test)
    infer_time = time.time() - start_inf

    # Scores
    metrics = {}
    if is_reg:
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        metrics.update({"R2": r2, "MSE": mse, "RMSE": rmse})
    else:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        metrics.update({"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec})

    # Cross-val (optional, on train only to save time)
    cv_mean = None
    if use_cv:
        try:
            if is_reg:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
            cv_mean = float(np.mean(cv_scores))
        except Exception:
            cv_mean = None

    return {
        "Model": name,
        **metrics,
        "Train Time (s)": train_time,
        "Inference Time (s)": infer_time,
        "CO2 Emissions (kg)": emissions,
        "CV (mean)": cv_mean,
        "preds": preds,
        "fitted_model": model
    }

results = []
plots = []

if go:
    if not selected_models:
        warn("Please select at least one model.")
    else:
        with st.spinner("Training selected models..."):
            zoo = build_model_zoo(task)
            for name in selected_models:
                mdl, is_reg = zoo[name]
                out = evaluate_model(name, mdl, is_reg)
                results.append(out)

        if results:
            # ---------------------------
            # Results Table
            # ---------------------------
            st.subheader("📊 Comparative Results")
            if task == "Regression":
                order_col = "R2"
                df_res = pd.DataFrame([
                    {k: v for k, v in r.items() if k in ["Model", "R2", "MSE", "RMSE", "Train Time (s)", "Inference Time (s)", "CO2 Emissions (kg)", "CV (mean)"]}
                    for r in results
                ]).sort_values(order_col, ascending=False)
            else:
                order_col = "Accuracy"
                df_res = pd.DataFrame([
                    {k: v for k, v in r.items() if k in ["Model", "Accuracy", "F1", "Precision", "Recall", "Train Time (s)", "Inference Time (s)", "CO2 Emissions (kg)", "CV (mean)"]}
                    for r in results
                ]).sort_values(order_col, ascending=False)

            st.dataframe(df_res, use_container_width=True)
            best_name = df_res.iloc[0]["Model"]
            success(f"Top performer by **{order_col}**: {best_name}")

            # ---------------------------
            # Diagnostics
            # ---------------------------
            st.subheader("🩺 Diagnostics")

            colA, colB = st.columns(2)

            if task == "Regression":
                # Residuals of best model
                best = next(r for r in results if r["Model"] == best_name)
                residuals = y_test - best["preds"]
                with colA:
                    st.markdown(f"**Residuals (Top Model: {best_name})**")
                    fig = plt.figure()
                    plt.hist(residuals, bins=50)
                    plt.xlabel("Residual")
                    plt.ylabel("Count")
                    plt.title("Residual Distribution")
                    st.pyplot(fig)

                # Scatter: y_true vs y_pred
                with colB:
                    st.markdown("**y_true vs y_pred (sample 10k)**")
                    samp = min(10_000, len(y_test))
                    idx = np.random.choice(len(y_test), samp, replace=False)
                    fig2 = plt.figure()
                    plt.scatter(np.array(y_test)[idx], np.array(best["preds"])[idx], s=6, alpha=0.5)
                    plt.xlabel("True")
                    plt.ylabel("Predicted")
                    plt.title("True vs Predicted")
                    st.pyplot(fig2)

            else:
                # Confusion matrix for best model
                best = next(r for r in results if r["Model"] == best_name)
                cm = confusion_matrix(y_test, best["preds"])
                with colA:
                    st.markdown(f"**Confusion Matrix (Top Model: {best_name})**")
                    fig = plt.figure()
                    plt.imshow(cm, interpolation="nearest")
                    plt.title("Confusion Matrix")
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.colorbar()
                    st.pyplot(fig)

                with colB:
                    st.markdown("**Class Distribution (y_test)**")
                    fig2 = plt.figure()
                    counts = pd.Series(y_test).value_counts().sort_index()
                    plt.bar(counts.index.astype(str), counts.values)
                    plt.xlabel("Class")
                    plt.ylabel("Count")
                    plt.title("Class Counts")
                    st.pyplot(fig2)

            # ---------------------------
            # Scenario: What-If Prediction
            # ---------------------------
            st.subheader("🧪 Scenario Tester (What-If)")

            # Prepare slider ranges from train data
            def range_for(col, pad=0.05):
                series = df[col] if col in df.columns else None
                if series is None or not np.issubdtype(series.dtype, np.number):
                    return None
                lo, hi = float(series.min()), float(series.max())
                span = hi - lo
                return (lo - pad * span, hi + pad * span)

            c1, c2, c3 = st.columns(3)
            rc_rng = range_for("RelativeCompactness");    sa_rng = range_for("SurfaceArea");   wa_rng = range_for("WallArea")
            ra_rng = range_for("RoofArea");               oh_rng = range_for("OverallHeight")

            with c1:
                rc = st.slider("RelativeCompactness", *(rc_rng if rc_rng else (0.5, 1.0)), 0.8)
                ori = st.selectbox("Orientation", ["North", "South", "East", "West"])
                btype = st.selectbox("BuildingType", ["Residential", "Commercial", "Industrial"])
            with c2:
                sa = st.slider("SurfaceArea", *(sa_rng if sa_rng else (50.0, 500.0)), 220.0)
                gad = st.selectbox("GlazingAreaDistribution", ["Uniform", "North-heavy", "South-heavy", "East-heavy", "West-heavy"])
                wa = st.slider("WallArea", *(wa_rng if wa_rng else (20.0, 300.0)), 130.0)
            with c3:
                ra = st.slider("RoofArea", *(ra_rng if ra_rng else (30.0, 200.0)), 95.0)
                oh = st.slider("OverallHeight", *(oh_rng if oh_rng else (2.5, 10.0)), 3.2)

            # Build one-row frame and one-hot like X columns
            scenario = pd.DataFrame([{
                'RelativeCompactness': rc, 'SurfaceArea': sa, 'WallArea': wa,
                'RoofArea': ra, 'OverallHeight': oh,
                'Orientation': ori, 'GlazingAreaDistribution': gad, 'BuildingType': btype
            }])

            scenario_X = pd.get_dummies(scenario, drop_first=True)
            # align to training columns
            for col in X_train.columns:
                if col not in scenario_X.columns:
                    scenario_X[col] = 0
            scenario_X = scenario_X[X_train.columns]

            # scale numeric if used
            if scale_numeric and scaler is not None:
                # Ensure columns match those used during scaler fitting
                scaler_cols = scaler.feature_names_in_
                for col in scaler_cols:
                    if col not in scenario_X.columns:
                        scenario_X[col] = 0
                scenario_X_scaled = scaler.transform(scenario_X[scaler_cols])
                scenario_X[scaler_cols] = scenario_X_scaled

            # Print the columns of scenario_X and the fitted feature names
            print("Columns in scenario_X:", scenario_X.columns.tolist())
            print("Fitted feature names:", scaler.get_feature_names_out())

            st.markdown("**Compare predictions across your selected models:**")
            rows = []
            for r in results:
                mdl = r["fitted_model"]
                try:
                    pred = mdl.predict(scenario_X)[0]
                except Exception:
                    pred = np.nan
                rows.append({"Model": r["Model"], "Prediction": float(pred) if np.isfinite(pred) else None})

            scen_df = pd.DataFrame(rows).sort_values("Model")
            st.dataframe(scen_df, use_container_width=True)

            if task == "Classification":
                st.caption("Predictions show the **class index** (0..N-1) for your chosen binning strategy.")

            st.markdown("---")

# ---------------------------
# Footer
# ---------------------------
st.caption("Tip: Start with subsampling (e.g., 200k rows) for speed, then scale up once you like the setup.")
