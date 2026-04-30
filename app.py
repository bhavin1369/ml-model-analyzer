"""
ML Model Workbench - Web Version
Flask-based web application for ML model training and comparison.
Deployable on Render free tier.
"""

import io
import os
import math
import json
import base64
import importlib
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# ── App Configuration ──────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ml-workbench-secret-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ── Model Builders ─────────────────────────────────────────────────────────────

def build_xgboost_regressor():
    try:
        xgb_module = importlib.import_module("xgboost")
        xgb_regressor_class = getattr(xgb_module, "XGBRegressor")
    except Exception:
        return None
    return xgb_regressor_class(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1,
    )


def build_lightgbm_regressor():
    try:
        lgbm_module = importlib.import_module("lightgbm")
        lgbm_regressor_class = getattr(lgbm_module, "LGBMRegressor")
    except Exception:
        return None
    return lgbm_regressor_class(
        n_estimators=350, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, verbosity=-1,
    )


MODEL_BUILDERS = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE),
    "Bagging": lambda: BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
        n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    "KNN": lambda: KNeighborsRegressor(n_neighbors=5),
    "SVR": lambda: SVR(kernel="rbf", C=50, epsilon=0.02),
    "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=250, random_state=RANDOM_STATE),
    "AdaBoost": lambda: AdaBoostRegressor(random_state=RANDOM_STATE),
}

# Try adding optional models
_xgb = build_xgboost_regressor()
if _xgb is not None:
    MODEL_BUILDERS["XGBoost"] = build_xgboost_regressor

_lgbm = build_lightgbm_regressor()
if _lgbm is not None:
    MODEL_BUILDERS["LightGBM"] = build_lightgbm_regressor

MODEL_INFO = {
    "Linear Regression": "Fast baseline linear model.",
    "Decision Tree": "Nonlinear tree model; easy to interpret.",
    "Random Forest": "Bagged trees; stable and strong for tabular data.",
    "Bagging": "Bootstrap aggregation with decision-tree base learners.",
    "Gradient Boosting": "Boosted trees; often strong accuracy.",
    "KNN": "Distance-based model; sensitive to scaling.",
    "SVR": "Support Vector Regression with RBF kernel.",
    "Extra Trees": "Randomized tree ensemble; robust and fast.",
    "AdaBoost": "Boosting model using weighted weak learners.",
    "LightGBM": "Histogram-based gradient boosting for fast, accurate tabular modeling.",
    "XGBoost": "Boosted tree ensemble with strong performance on tabular datasets.",
}


# ── Helper Functions ───────────────────────────────────────────────────────────

def load_dataset(path_str: str) -> pd.DataFrame:
    csv_path = Path(path_str).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one target column.")
    df = df.dropna(axis=0).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Dataset has too few valid rows after cleaning (need at least 10).")
    return df


def format_feature_value(value):
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 10000 or (0 < abs(value) < 0.001):
            return f"{value:.2e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def train_models(df, selected_models, test_size=0.2):
    """Train selected models and return metrics, predictions, and top combinations."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE,
    )

    records = []
    predictions = {}
    top_combinations = {}

    for model_name in selected_models:
        builder = MODEL_BUILDERS.get(model_name)
        if builder is None:
            continue

        model = builder()
        if model is None:
            continue

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        records.append({
            "Model": model_name,
            "RMSE": round(rmse, 6),
            "MSE": round(mse, 6),
            "R2": round(r2, 6),
        })

        pred_df = pd.DataFrame({
            "Actual": y_test.reset_index(drop=True),
            "Predicted": pd.Series(y_pred),
        })
        predictions[model_name] = pred_df

        # Top 10 combinations
        combo_features = X.drop_duplicates().reset_index(drop=True)
        combo_preds = pipeline.predict(combo_features)
        combo_df = combo_features.copy()
        combo_df["Predicted"] = combo_preds
        combo_df = combo_df.sort_values("Predicted", ascending=False).head(10).reset_index(drop=True)
        top_combinations[model_name] = combo_df

    metrics_df = pd.DataFrame(records).sort_values("R2", ascending=False).reset_index(drop=True)
    return metrics_df, predictions, top_combinations, feature_cols


def _viridis_colors(n):
    """Generate n colors from viridis-like rainbow palette for research papers."""
    palette = [
        '#440154', '#482878', '#3e4989', '#31688e', '#26828e',
        '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725',
    ]
    if n <= len(palette):
        return palette[:n]
    out = []
    for i in range(n):
        idx = i * (len(palette) - 1) / max(n - 1, 1)
        out.append(palette[min(int(round(idx)), len(palette) - 1)])
    return out


# 10 distinct research-paper scatter colors (colorblind-friendly)
SCATTER_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff',
]


def build_comparison_chart(metrics_df):
    """Build model comparison chart — research paper quality."""
    if metrics_df.empty:
        return "{}"

    display_df = metrics_df.sort_values("R2", ascending=True).reset_index(drop=True)
    models = display_df["Model"].tolist()
    r2_vals = (display_df["R2"] * 100).tolist()
    rmse_vals = display_df["RMSE"].tolist()
    best_model = display_df.iloc[-1]
    n = len(models)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"<b>R² Accuracy Comparison</b><br><sup>Best: {best_model['Model']} ({best_model['R2']*100:.2f}%)</sup>",
            f"<b>RMSE Error Comparison</b><br><sup>Best: {best_model['Model']} ({best_model['RMSE']:.6f})</sup>",
        ],
        horizontal_spacing=0.18,
    )

    colors_r2 = _viridis_colors(n)
    colors_r2[-1] = '#e6194b'  # highlight best
    fig.add_trace(go.Bar(
        y=models, x=r2_vals, orientation='h',
        marker=dict(color=colors_r2, line=dict(color='#333', width=0.5)),
        text=[f"{v:.2f}%" for v in r2_vals], textposition='outside',
        textfont=dict(size=10), name='R²',
    ), row=1, col=1)

    colors_rmse = _viridis_colors(n)
    colors_rmse[-1] = '#e6194b'
    fig.add_trace(go.Bar(
        y=models, x=rmse_vals, orientation='h',
        marker=dict(color=colors_rmse, line=dict(color='#333', width=0.5)),
        text=[f"{v:.6f}" for v in rmse_vals], textposition='outside',
        textfont=dict(size=10), name='RMSE',
    ), row=1, col=2)

    fig.update_layout(
        height=max(420, n * 48 + 140), showlegend=False,
        font=dict(family="Inter, serif", size=12, color='#222'),
        paper_bgcolor='#fff', plot_bgcolor='#fff',
        margin=dict(l=10, r=50, t=80, b=40),
    )
    for col_idx in [1, 2]:
        fig.update_xaxes(
            showgrid=True, gridcolor='#e0e0e0', gridwidth=1,
            zeroline=True, zerolinecolor='#ccc',
            row=1, col=col_idx,
        )
        fig.update_yaxes(
            showgrid=False, tickfont=dict(size=11),
            row=1, col=col_idx,
        )
    fig.update_xaxes(title_text="R² Score (%)", title_font=dict(size=12), row=1, col=1)
    fig.update_xaxes(title_text="RMSE", title_font=dict(size=12), row=1, col=2)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def build_prediction_chart_single(name, frame, metric_row, color_idx=0):
    """Build a single predicted-vs-actual chart for one model — research paper quality."""
    actual = frame["Actual"].values
    predicted = frame["Predicted"].values
    residual = actual - predicted
    all_vals = np.concatenate([actual, predicted])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.05 * (hi - lo if hi != lo else 1.0)
    lo, hi = lo - pad, hi + pad

    color = SCATTER_COLORS[color_idx % len(SCATTER_COLORS)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(
            size=5, color=residual, colorscale='RdYlGn_r',
            showscale=True, colorbar=dict(title='Residual', thickness=14, len=0.8),
            line=dict(width=0.3, color='#555'), opacity=0.85,
        ),
        name='Data points',
        hovertemplate='Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode='lines',
        line=dict(color='#c0392b', dash='dash', width=2),
        name='Perfect prediction (y=x)',
    ))

    r2 = metric_row['R2']
    rmse = metric_row['RMSE']
    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b><br><sup>R² = {r2:.4f} | RMSE = {rmse:.6f}</sup>",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Actual Values", range=[lo, hi],
            showgrid=True, gridcolor='#e8e8e8', zeroline=False,
        ),
        yaxis=dict(
            title="Predicted Values", range=[lo, hi],
            showgrid=True, gridcolor='#e8e8e8', zeroline=False,
            scaleanchor="x", scaleratio=1,
        ),
        height=480, width=560,
        paper_bgcolor='#fff', plot_bgcolor='#fafafa',
        font=dict(family="Inter, serif", size=11, color='#222'),
        margin=dict(l=60, r=20, t=70, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', font=dict(size=10)),
        showlegend=True,
    )
    return fig


def build_prediction_charts(metrics_df, predictions):
    """Build per-model predicted vs actual charts — returns dict of model_name -> plotly JSON."""
    if metrics_df.empty:
        return {}

    result = {}
    for idx, name in enumerate(metrics_df["Model"].tolist()):
        if name not in predictions:
            continue
        frame = predictions[name]
        metric_row = metrics_df[metrics_df["Model"] == name].iloc[0]
        fig = build_prediction_chart_single(name, frame, metric_row, idx)
        result[name] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return result


def build_combination_chart_single(name, top_df, feature_cols):
    """Build a single top-10 combinations chart for one model — research paper quality with viridis bars."""
    top_df = top_df.sort_values("Predicted", ascending=False).reset_index(drop=True)
    labels = [f"C{i+1}" for i in range(len(top_df))]
    values = top_df["Predicted"].tolist()
    n = len(values)
    colors = _viridis_colors(n)

    hover_texts = []
    for _, row in top_df.iterrows():
        parts = [f"{col}: {format_feature_value(row[col])}" for col in feature_cols if col in row.index]
        hover_texts.append("<br>".join(parts) + f"<br><b>Predicted: {row['Predicted']:.4f}</b>")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color='#333', width=0.5)),
        text=[f"{v:.4f}" for v in values], textposition='outside',
        textfont=dict(size=10),
        hovertext=hover_texts, hoverinfo='text',
    ))

    ymin = min(values) if values else 0
    ymax = max(values) if values else 1
    ypad = (ymax - ymin) * 0.12 if ymax != ymin else 0.02
    fig.update_layout(
        title=dict(
            text=f"<b>{name} — Top 10 Best Combinations</b>",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Combination Rank",
            showgrid=False,
        ),
        yaxis=dict(
            title="Predicted Target Value",
            showgrid=True, gridcolor='#e8e8e8',
            range=[ymin - ypad, ymax + ypad],
        ),
        height=460, width=620,
        paper_bgcolor='#fff', plot_bgcolor='#fafafa',
        font=dict(family="Inter, serif", size=11, color='#222'),
        margin=dict(l=60, r=20, t=60, b=50),
        showlegend=False,
    )
    return fig


def build_combination_charts(metrics_df, top_combinations, feature_cols):
    """Build per-model top-10 combination charts — returns dict of model_name -> plotly JSON."""
    if not top_combinations:
        return {}

    result = {}
    for name in metrics_df["Model"].tolist():
        if name not in top_combinations:
            continue
        fig = build_combination_chart_single(name, top_combinations[name], feature_cols)
        result[name] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return result


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        models=list(MODEL_BUILDERS.keys()),
        model_info=MODEL_INFO,
    )


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        "models": list(MODEL_BUILDERS.keys()),
        "info": {k: MODEL_INFO.get(k, "") for k in MODEL_BUILDERS},
    })


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(str(filepath))

    try:
        df = load_dataset(str(filepath))
        info = {
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "features": df.columns[:-1].tolist(),
            "target": df.columns[-1],
            "filepath": str(filepath),
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/train", methods=["POST"])
def train():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    filepath = data.get("filepath")
    selected_models = data.get("models", [])
    test_size = data.get("test_size", 0.2)

    if not filepath:
        return jsonify({"error": "No dataset specified"}), 400
    if not selected_models:
        return jsonify({"error": "No models selected"}), 400

    try:
        df = load_dataset(filepath)
        metrics_df, predictions, top_combinations, feature_cols = train_models(
            df, selected_models, test_size
        )

        # Build charts
        comparison_chart = build_comparison_chart(metrics_df)
        prediction_chart = build_prediction_charts(metrics_df, predictions)
        combination_chart = build_combination_charts(metrics_df, top_combinations, feature_cols)

        # Build results table
        results_table = metrics_df.to_dict(orient="records")

        # Build top combinations detail
        combo_details = {}
        for model_name, combo_df in top_combinations.items():
            combo_details[model_name] = combo_df.round(4).to_dict(orient="records")

        # Best model summary
        best = metrics_df.iloc[0]
        summary = f"Best model: {best['Model']} (R²={best['R2']:.4f}, RMSE={best['RMSE']:.4f})"

        return jsonify({
            "summary": summary,
            "results_table": results_table,
            "comparison_chart": comparison_chart,
            "prediction_chart": prediction_chart,
            "combination_chart": combination_chart,
            "combo_details": combo_details,
            "feature_cols": feature_cols,
            "dataset_info": {
                "rows": len(df),
                "features": feature_cols,
                "target": df.columns[-1],
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset-info", methods=["POST"])
def dataset_info():
    data = request.get_json()
    filepath = data.get("filepath")
    if not filepath:
        return jsonify({"error": "No filepath"}), 400
    try:
        df = load_dataset(filepath)
        return jsonify({
            "rows": len(df),
            "columns": len(df.columns),
            "features": df.columns[:-1].tolist(),
            "target": df.columns[-1],
            "preview": df.head(5).to_html(classes="table table-sm", index=False),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
