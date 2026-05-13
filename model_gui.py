import math
import importlib
import textwrap
import joblib
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
DEFAULT_DATASET = Path("data.csv")
LIGHTGBM_FALLBACK_MESSAGE = None
XGBOOST_FALLBACK_MESSAGE = None
DEFAULT_RATIO_LIST_TEXT = "80,70,60"
PREDICTION_PLOT_MIN = 0.0
PREDICTION_PLOT_MAX = 1.0

QUICK_START_GUIDE_TEXT = """QUICK START GUIDE

1. LOAD YOUR DATA
    - Click \"Select CSV File\"
    - Choose your CSV file
    - File must have headers (column names)
    - Last column should be your target output

2. SELECT MODELS
    - Choose which models to train
    - Recommended: Use all models
    - Different models work differently

3. TRAIN MODELS
    - Click \"TRAIN MODELS\"
    - Wait for training to complete
    - Check PROGRESS section

4. VIEW RESULTS
    - Go to \"Summary\" tab to see results
    - Go to \"Metrics Table\" to see detailed scores
    - Go to \"Charts\" to see visualizations

===============================================

UNDERSTANDING THE METRICS

R2 Score (Accuracy):
    1.00: Perfect prediction
    0.99 - 1.00: EXCELLENT (near perfect)
    0.90 - 0.99: VERY GOOD (excellent accuracy)
    0.75 - 0.90: GOOD (very useful)
    0.50 - 0.75: FAIR (acceptable)
    < 0.50: POOR (not very useful)

RMSE (Error):
    - Lower is better
    - Shows average prediction error
    - Same units as your target variable

===============================================

ABOUT THE MODELS

Extra Trees: Fast ensemble method
Random Forest: Popular and reliable
XGBoost: Powerful gradient boosting
AdaBoost: Boosts weak learners
Linear Regression: Simple baseline
SVR RBF: Works with complex patterns
KNN k5: Simple neighbor-based method

===============================================

TIPS FOR BEST RESULTS

Good practices:
    - Use at least 100-200 samples in your data
    - Ensure your target variable is numerical
    - More data = usually better predictions
    - Try different model combinations
    - Compare R2 scores to find best model

Things to avoid:
    - Very small datasets (less than 50 samples)
    - Non-numerical target variables
    - Data with too many missing values
    - Extreme outliers"""


def build_xgboost_regressor():
    global XGBOOST_FALLBACK_MESSAGE
    try:
        xgb_module = importlib.import_module("xgboost")
        xgb_regressor_class = getattr(xgb_module, "XGBRegressor")
        XGBOOST_FALLBACK_MESSAGE = None
    except (ImportError, AttributeError):
        XGBOOST_FALLBACK_MESSAGE = (
            "xgboost is not available in this Python environment. Using HistGradientBoostingRegressor as a local fallback so runs can continue."
        )
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=350,
            max_depth=6,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )

    return xgb_regressor_class(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_lightgbm_regressor():
    global LIGHTGBM_FALLBACK_MESSAGE
    try:
        lgbm_module = importlib.import_module("lightgbm")
        lgbm_regressor_class = getattr(lgbm_module, "LGBMRegressor")
        LIGHTGBM_FALLBACK_MESSAGE = None
        return lgbm_regressor_class(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    except (ImportError, AttributeError):
        LIGHTGBM_FALLBACK_MESSAGE = (
            "LightGBM is not installed. Using HistGradientBoostingRegressor as a local fallback so runs can continue."
        )
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=350,
            max_depth=6,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )

MODEL_BUILDERS = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE),
    "Bagging": lambda: BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    "KNN": lambda: KNeighborsRegressor(n_neighbors=5),
    "SVR": lambda: SVR(kernel="rbf", C=50, epsilon=0.02),
    "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=250, random_state=RANDOM_STATE),
    "AdaBoost": lambda: AdaBoostRegressor(random_state=RANDOM_STATE),
    "LightGBM": build_lightgbm_regressor,
    "XGBoost": build_xgboost_regressor,
}

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


class ModelWorkbenchGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ML Model Workbench")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 700)

        self.dataset_path = tk.StringVar(value=str(DEFAULT_DATASET))
        self.train_ratio_var = tk.StringVar(value="80")
        self.test_ratio_var = tk.StringVar(value="20")
        self.ratio_list_var = tk.StringVar(value=DEFAULT_RATIO_LIST_TEXT)
        self.target_col_var = tk.StringVar(value="")
        self._ratio_sync_in_progress = False
        self.train_size_value = 0.8
        self.test_size_value = 0.2
        self.status_text = tk.StringVar(value="Select dataset and model(s), then click Run Selected Models.")

        self.metrics_df = pd.DataFrame()
        self.ratio_metrics_df = pd.DataFrame()
        self.ratio_best_r2_row = None
        self.ratio_best_rmse_row = None
        self.ratio_best_mse_row = None
        self.ratio_best_mae_row = None
        self.dataset_top10 = None
        self.dataset_target_col = ""
        self.prediction_frames = {}
        self.top_combinations = {}
        self.prediction_filter_frame = None
        self.combination_filter_frame = None
        self.prediction_view_vars = {}
        self.combination_view_vars = {}
        self.model_select_vars = {}
        self.comparison_mode_var = tk.StringVar(value="2D")
        self.ratio_mode_var = tk.StringVar(value="2D")
        self.prediction_axis_auto_var = tk.BooleanVar(value=True)
        self.prediction_xmin_var = tk.StringVar(value="")
        self.prediction_xmax_var = tk.StringVar(value="")
        self.prediction_ymin_var = tk.StringVar(value="")
        self.prediction_ymax_var = tk.StringVar(value="")
        self.prediction_xunit_var = tk.StringVar(value="")
        self.prediction_yunit_var = tk.StringVar(value="")
        self.ratio_model_var = tk.StringVar(value="")
        self.trained_models = {}
        self.progress_var = tk.DoubleVar(value=0.0)
        self._canvas_interactions = {}

        self._setup_matplotlib_defaults()
        self._configure_style()
        self._build_layout()
        self._populate_model_list()

    # ----- Interactive canvas helpers -----
    def _enable_canvas_interactions(self, canvas):
        state = {"annotation": None, "selectors": [], "default_limits": {}, "connections": {}, "hover_artists": []}
        self._canvas_interactions[canvas] = state

        def on_move(event, target_canvas=canvas):
            self._handle_hover_event(target_canvas, event)

        def on_leave(_event, target_canvas=canvas):
            self._hide_hover_annotation(target_canvas)

        def on_click(event, target_canvas=canvas):
            # right-click resets zoom
            if getattr(event, "button", None) == 3:
                self._reset_zoom(target_canvas, event.inaxes)

        state["connections"]["motion"] = canvas.mpl_connect("motion_notify_event", on_move)
        state["connections"]["leave"] = canvas.mpl_connect("figure_leave_event", on_leave)
        state["connections"]["click"] = canvas.mpl_connect("button_press_event", on_click)

    def _set_hover_data(self, artist, text=None, texts=None):
        if text is not None:
            setattr(artist, "_hover_text", text)
        if texts is not None:
            setattr(artist, "_hover_texts", list(texts))
        # register artist to canvas hover list
        try:
            fig = artist.figure
            for canvas, state in self._canvas_interactions.items():
                if getattr(canvas, "figure", None) is fig:
                    if artist not in state["hover_artists"]:
                        state["hover_artists"].append(artist)
                    break
        except Exception:
            pass

    def _find_hover_text(self, canvas, event):
        ax = event.inaxes
        if ax is None:
            return None
        state = self._canvas_interactions.get(canvas)
        if state is None:
            return None
        artists = [a for a in state.get("hover_artists", []) if getattr(a, "axes", None) is ax]
        for artist in reversed(artists):
            has_single = hasattr(artist, "_hover_text")
            has_multi = hasattr(artist, "_hover_texts")
            try:
                contains, info = artist.contains(event)
            except Exception:
                continue
            if not contains:
                continue
            if has_multi:
                indices = info.get("ind", []) if isinstance(info, dict) else []
                if len(indices) > 0:
                    texts = getattr(artist, "_hover_texts", [])
                    idx = int(indices[0])
                    if 0 <= idx < len(texts):
                        return texts[idx]
                continue
            return getattr(artist, "_hover_text", None)
        return None

    def _handle_hover_event(self, canvas, event):
        state = self._canvas_interactions.get(canvas)
        if state is None:
            return
        if event.inaxes is None:
            self._hide_hover_annotation(canvas)
            return

        # Selection cursor: use arrow for 'Auto' mode, crosshair for manual selection/zoom
        if canvas == getattr(self, "pred_canvas", None) and self.prediction_axis_auto_var.get():
            canvas.get_tk_widget().configure(cursor="arrow")
        else:
            canvas.get_tk_widget().configure(cursor="crosshair")

        text = self._find_hover_text(canvas, event)
        if not text:
            self._hide_hover_annotation(canvas)
            return
        if state["annotation"] is None or state["annotation"].axes is not event.inaxes:
            self._hide_hover_annotation(canvas)
            state["annotation"] = event.inaxes.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#135d66", edgecolor="#0b3f45", alpha=0.95),
                color="white",
                fontsize=9,
                ha="left",
                va="bottom",
            )
        annotation = state["annotation"]
        x = event.xdata if event.xdata is not None else 0.0
        y = event.ydata if event.ydata is not None else 0.0
        annotation.xy = (x, y)
        if annotation.get_text() != str(text) or not annotation.get_visible():
            annotation.set_text(str(text))
            annotation.set_visible(True)
            canvas.draw_idle()

    def _hide_hover_annotation(self, canvas):
        state = self._canvas_interactions.get(canvas)
        if state is None:
            return
        # Reset cursor to default
        try:
            canvas.get_tk_widget().configure(cursor="")
        except Exception:
            pass
        annotation = state.get("annotation")
        if annotation is not None and annotation.get_visible():
            annotation.set_visible(False)
            canvas.draw_idle()

    def _setup_canvas_zoom_selectors(self, canvas):
        state = self._canvas_interactions.get(canvas)
        if state is None:
            return
        # clear any existing selectors
        for selector in state.get("selectors", []):
            try:
                selector.set_active(False)
            except Exception:
                pass
        state["selectors"] = []
        state["default_limits"] = {}
        axes_2d = [ax for ax in canvas.figure.axes if getattr(ax, "name", "") != "3d"]
        for ax in axes_2d:
            state["default_limits"][id(ax)] = (ax.get_xlim(), ax.get_ylim())

            def on_select(eclick, erelease, target_ax=ax, target_canvas=canvas):
                if eclick.xdata is None or erelease.xdata is None or eclick.ydata is None or erelease.ydata is None:
                    return
                x0, x1 = sorted([eclick.xdata, erelease.xdata])
                y0, y1 = sorted([eclick.ydata, erelease.ydata])
                if abs(x1 - x0) <= 1e-12 or abs(y1 - y0) <= 1e-12:
                    return
                target_ax.set_xlim(x0, x1)
                target_ax.set_ylim(y0, y1)
                target_canvas.draw_idle()

            selector = RectangleSelector(
                ax,
                on_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=False,
                drag_from_anywhere=False,
            )
            state["selectors"].append(selector)

    def _reset_canvas_hover_state(self, canvas):
        state = self._canvas_interactions.get(canvas)
        if state is None:
            return
        annotation = state.get("annotation")
        if annotation is not None:
            try:
                annotation.remove()
            except Exception:
                pass
        state["annotation"] = None
        state["hover_artists"] = []

    def _reset_zoom(self, canvas, event_ax=None):
        state = self._canvas_interactions.get(canvas)
        if state is None:
            return
        changed = False
        for ax in canvas.figure.axes:
            if getattr(ax, "name", "") == "3d":
                continue
            if event_ax is not None and ax is not event_ax:
                continue
            default = state["default_limits"].get(id(ax))
            if default is None:
                continue
            xlim, ylim = default
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            changed = True
        if changed:
            canvas.draw_idle()

    def _build_graph_section(self, parent, title, fullscreen_command):
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x", padx=8, pady=(8, 0))

        ttk.Label(header, text=title, style="Subtitle.TLabel").pack(side="left")

        right_controls = ttk.Frame(header)
        right_controls.pack(side="right")

        filter_frame = ttk.Frame(right_controls)
        filter_frame.pack(side="left", padx=(0, 8))

        ttk.Button(right_controls, text="Full Screen", command=fullscreen_command).pack(side="left")

        toolbar_holder = ttk.Frame(container, padding=(8, 0, 8, 4))
        toolbar_holder.pack(fill="x")

        canvas_holder = ttk.Frame(container, padding=(8, 2, 8, 8))
        canvas_holder.pack(fill="both", expand=True)
        return container, canvas_holder, filter_frame, toolbar_holder

    def _add_canvas_toolbar(self, parent, canvas):
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left")

        ttk.Button(parent, text="Reset Zoom", command=lambda: self._reset_zoom(canvas)).pack(side="left", padx=(8, 0))
        ttk.Label(parent, text="Left drag: zoom | Right click: reset").pack(side="left", padx=(12, 0))

    def _build_comparison_mode_controls(self, parent):
        for child in parent.winfo_children():
            child.destroy()
        ttk.Label(parent, text="View:").pack(side="left", padx=(0, 6))

        for mode in ("2D", "3D"):
            ttk.Radiobutton(
                parent,
                text=mode,
                value=mode,
                variable=self.comparison_mode_var,
                command=self._on_comparison_mode_change,
            ).pack(side="left", padx=(0, 4))

    def _build_ratio_mode_controls(self, parent):
        for child in parent.winfo_children():
            child.destroy()
        
        ttk.Label(parent, text="Model:").pack(side="left", padx=(0, 4))
        self.ratio_model_combo = ttk.Combobox(parent, textvariable=self.ratio_model_var, state="readonly", width=18)
        self.ratio_model_combo.pack(side="left", padx=(0, 10))
        self.ratio_model_combo.bind("<<ComboboxSelected>>", lambda e: self._on_ratio_mode_change())

        ttk.Label(parent, text="View:").pack(side="left", padx=(0, 6))

        for mode in ("2D", "3D"):
            ttk.Radiobutton(
                parent,
                text=mode,
                value=mode,
                variable=self.ratio_mode_var,
                command=self._on_ratio_mode_change,
            ).pack(side="left", padx=(0, 4))

    def _build_prediction_axis_controls(self, parent):
        for child in parent.winfo_children():
            child.destroy()

        # Use tk.Checkbutton for native checkmark symbol
        chk = tk.Checkbutton(
            parent,
            text="Auto from dataset",
            variable=self.prediction_axis_auto_var,
            command=self._on_prediction_axis_change,
            anchor="w",
            relief="flat",
        )
        chk.grid(row=0, column=0, columnspan=4, sticky="w", padx=(2, 6), pady=(0, 6))

        ttk.Label(parent, text="X min:").grid(row=1, column=0, sticky="e", padx=(2, 4))
        ttk.Entry(parent, textvariable=self.prediction_xmin_var, width=10).grid(row=1, column=1, sticky="w", padx=(0, 8))
        ttk.Label(parent, text="X max:").grid(row=1, column=2, sticky="e", padx=(2, 4))
        ttk.Entry(parent, textvariable=self.prediction_xmax_var, width=10).grid(row=1, column=3, sticky="w", padx=(0, 4))

        ttk.Label(parent, text="Y min:").grid(row=2, column=0, sticky="e", padx=(2, 4), pady=(4, 0))
        ttk.Entry(parent, textvariable=self.prediction_ymin_var, width=10).grid(row=2, column=1, sticky="w", padx=(0, 8), pady=(4, 0))
        ttk.Label(parent, text="Y max:").grid(row=2, column=2, sticky="e", padx=(2, 4), pady=(4, 0))
        ttk.Entry(parent, textvariable=self.prediction_ymax_var, width=10).grid(row=2, column=3, sticky="w", padx=(0, 4), pady=(4, 0))

        ttk.Label(parent, text="X unit:").grid(row=3, column=0, sticky="e", padx=(2, 4), pady=(4, 0))
        ttk.Entry(parent, textvariable=self.prediction_xunit_var, width=10).grid(row=3, column=1, sticky="w", padx=(0, 8), pady=(4, 0))
        ttk.Label(parent, text="Y unit:").grid(row=3, column=2, sticky="e", padx=(2, 4), pady=(4, 0))
        ttk.Entry(parent, textvariable=self.prediction_yunit_var, width=10).grid(row=3, column=3, sticky="w", padx=(0, 4), pady=(4, 0))

        ttk.Button(parent, text="Apply Axis", command=self._on_prediction_axis_change).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=(2, 8), pady=(6, 0)
        )
        ttk.Label(parent, text="Applies to all Predicted vs Actual plots").grid(
            row=4, column=2, columnspan=2, sticky="w", padx=(4, 0), pady=(6, 0)
        )

        # allow the view holder to expand while keeping axis controls compact
        for c in range(4):
            try:
                parent.grid_columnconfigure(c, weight=0)
            except Exception:
                pass

    def _infer_prediction_units_from_target(self, target_col):
        colname = (target_col or "").lower()
        if "thz" in colname:
            return "THz"
        if "nm" in colname:
            return "nm"
        if "db" in colname:
            return "dB"
        if "hz" in colname:
            return "Hz"
        if "%" in colname:
            return "%"
        return ""

    def _sync_prediction_units_from_target(self):
        unit = self._infer_prediction_units_from_target(self.dataset_target_col or self.target_col_var.get())
        if unit:
            if not self.prediction_xunit_var.get().strip():
                self.prediction_xunit_var.set(unit)
            if not self.prediction_yunit_var.get().strip():
                self.prediction_yunit_var.set(unit)

    def _parse_axis_limit(self, raw_value):
        text = str(raw_value).strip()
        if text == "":
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return value

    def _get_prediction_axis_limits(self, frames):
        actual_values = [frame["Actual"].to_numpy(dtype=float) for frame in frames]
        predicted_values = [frame["Predicted"].to_numpy(dtype=float) for frame in frames]
        combined_values = np.concatenate(actual_values + predicted_values)

        combined_min = float(np.nanmin(combined_values))
        combined_max = float(np.nanmax(combined_values))
        if math.isfinite(combined_min) and math.isfinite(combined_max):
            span = combined_max - combined_min
            if span == 0 or np.isclose(span, 0.0):
                pad = abs(combined_max) * 0.05 if combined_max != 0 else 0.5
            else:
                pad = span * 0.04
            auto_limits = (combined_min - pad, combined_max + pad)
        else:
            auto_limits = (PREDICTION_PLOT_MIN, PREDICTION_PLOT_MAX)

        if self.prediction_axis_auto_var.get():
            return (*auto_limits, *auto_limits)

        x_min = self._parse_axis_limit(self.prediction_xmin_var.get())
        x_max = self._parse_axis_limit(self.prediction_xmax_var.get())
        y_min = self._parse_axis_limit(self.prediction_ymin_var.get())
        y_max = self._parse_axis_limit(self.prediction_ymax_var.get())

        x_limits = auto_limits if x_min is None or x_max is None or x_min >= x_max else (x_min, x_max)
        y_limits = auto_limits if y_min is None or y_max is None or y_min >= y_max else (y_min, y_max)
        return (*x_limits, *y_limits)

    def _on_prediction_axis_change(self, _event=None):
        self._reset_canvas_hover_state(self.pred_canvas)
        self._render_prediction_figure(self.pred_fig)
        self._setup_canvas_zoom_selectors(self.pred_canvas)
        self.pred_canvas.draw_idle()

    def _schedule_axis_update(self):
        # debounce quick successive edits
        try:
            if getattr(self, "_axis_update_after_id", None):
                try:
                    self.root.after_cancel(self._axis_update_after_id)
                except Exception:
                    pass
            self._axis_update_after_id = self.root.after(250, self._on_prediction_axis_change)
        except Exception:
            # fallback immediate call
            try:
                self._on_prediction_axis_change()
            except Exception:
                pass

    def _bind_prediction_axis_traces(self):
        # trace write on vars to schedule updates
        try:
            vars_to_bind = [
                self.prediction_xmin_var,
                self.prediction_xmax_var,
                self.prediction_ymin_var,
                self.prediction_ymax_var,
                self.prediction_xunit_var,
                self.prediction_yunit_var,
                self.prediction_axis_auto_var,
            ]
            for v in vars_to_bind:
                try:
                    v.trace_add("write", lambda *a: self._schedule_axis_update())
                except Exception:
                    try:
                        # older tkinter fallback
                        v.trace("w", lambda *a: self._schedule_axis_update())
                    except Exception:
                        pass
        except Exception:
            pass

    def _create_view_checkboxes(self, parent, view_vars, callback):
        for child in parent.winfo_children():
            child.destroy()

        ttk.Label(parent, text="View:").grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 6))

        all_var = tk.BooleanVar(value=False)
        view_vars.clear()
        view_vars["All Models"] = all_var

        def on_all_toggle():
            if all_var.get():
                for name, var in view_vars.items():
                    if name != "All Models":
                        var.set(False)
            else:
                for name in MODEL_BUILDERS.keys():
                    view_vars[name].set(True)
            self._sync_view_state(view_vars)
            if callback is not None:
                callback()

        tk.Checkbutton(
            parent,
            text="All Models",
            variable=all_var,
            command=on_all_toggle,
            anchor="w",
            relief="flat",
        ).grid(row=0, column=1, sticky="w", padx=(0, 6))

        model_frame = ttk.Frame(parent)
        model_frame.grid(row=1, column=1, columnspan=3, sticky="w")
        parent.grid_columnconfigure(2, weight=1)

        self._populate_view_checkboxes(model_frame, view_vars, callback)
        self._sync_view_state(view_vars)

    def _populate_view_checkboxes(self, parent, view_vars, callback):
        for child in parent.winfo_children():
            child.destroy()

        for idx, model_name in enumerate(MODEL_BUILDERS.keys()):
            var = tk.BooleanVar(value=True)
            view_vars[model_name] = var

            def on_toggle(name=model_name):
                view_vars["All Models"].set(all(view_vars[name].get() for name in MODEL_BUILDERS.keys()))
                self._sync_view_state(view_vars)
                if callback is not None:
                    callback()

            tk.Checkbutton(
                parent,
                text=model_name,
                variable=var,
                command=on_toggle,
                anchor="w",
                relief="flat",
            ).grid(
                row=idx // 4,
                column=idx % 4,
                sticky="w",
                padx=(0, 10),
                pady=1,
            )

    def _sync_view_state(self, view_vars):
        if view_vars["All Models"].get():
            return

    def _get_selected_models(self, view_vars, available_models):
        if view_vars.get("All Models") is not None and view_vars["All Models"].get():
            return available_models

        selected = []
        for name in available_models:
            var = view_vars.get(name)
            if var is not None and var.get():
                selected.append(name)
        return selected

    def _setup_matplotlib_defaults(self):
        """Sets global Matplotlib parameters for professional, research-ready visuals."""
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "#ffffff",
            "savefig.dpi": 300,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False
        })

    def _configure_style(self):
        style = ttk.Style()
        # Use native theme where possible
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "xpnative" in style.theme_names():
            style.theme_use("xpnative")

        # Professional Color Palette
        bg_main = "#f8fafc"  # Very light grey-blue
        accent_blue = "#1e40af" # Deep blue
        
        style.configure("TFrame", background=bg_main)
        style.configure("TLabel", background=bg_main, font=("Segoe UI", 10))
        
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=accent_blue)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10, "bold"), foreground="#334155")
        style.configure("Info.TLabel", font=("Segoe UI", 9), foreground="#64748b")
        
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        
        style.configure("Control.TLabelframe", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Control.TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#1e293b")
        
        style.configure("ControlHeading.TLabel", font=("Segoe UI", 14, "bold"), foreground="#0f172a")
        
        # Treeview (Results Table)
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        
        style.configure("Horizontal.TProgressbar", thickness=10)

    def _build_layout(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="ML Model Workbench", style="Title.TLabel").grid(row=0, column=0, sticky="w", columnspan=6)

        ttk.Label(top, text="Dataset (CSV)", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 2))
        entry = ttk.Entry(top, textvariable=self.dataset_path, width=75)
        entry.grid(row=2, column=0, columnspan=4, sticky="we", padx=(0, 8))

        ttk.Button(top, text="Browse", command=self._browse_dataset).grid(row=2, column=4, sticky="we", padx=(0, 6))
        ttk.Button(top, text="Load Info", command=self._show_dataset_info).grid(row=2, column=5, sticky="we")

        ttk.Label(top, text="Target Column", style="Subtitle.TLabel").grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.target_col_combo = ttk.Combobox(top, textvariable=self.target_col_var, width=40, state="readonly")
        self.target_col_combo.grid(row=4, column=0, columnspan=2, sticky="we", padx=(0, 8))
        self.target_col_combo.bind("<<ComboboxSelected>>", self._on_target_col_change)

        ttk.Label(top, text="Train Ratio (%)", style="Subtitle.TLabel").grid(row=5, column=0, sticky="w", pady=(8, 2))
        train_entry = ttk.Entry(top, textvariable=self.train_ratio_var, width=12)
        train_entry.grid(row=6, column=0, sticky="w")

        ttk.Label(top, text="Test Ratio (%)", style="Subtitle.TLabel").grid(row=5, column=1, sticky="w", pady=(8, 2))
        test_entry = ttk.Entry(top, textvariable=self.test_ratio_var, width=12)
        test_entry.grid(row=6, column=1, sticky="w")

        

        ttk.Label(top, text="Train Ratios List (%)", style="Subtitle.TLabel").grid(row=5, column=2, sticky="w", pady=(8, 2))
        ratios_entry = ttk.Entry(top, textvariable=self.ratio_list_var, width=28)
        ratios_entry.grid(row=6, column=2, columnspan=2, sticky="we", padx=(0, 8))

        ttk.Label(top, text="Default: 80,70,60", style="Info.TLabel").grid(row=6, column=4, sticky="w")

        ttk.Button(top, text="Run Selected Models", style="Accent.TButton", command=self.run_selected_models).grid(
            row=6, column=5, sticky="we", pady=(6, 0)
        )

        train_entry.bind("<KeyRelease>", self._on_train_ratio_change)
        train_entry.bind("<FocusOut>", self._on_train_ratio_focus_out)
        test_entry.bind("<KeyRelease>", self._on_test_ratio_change)
        test_entry.bind("<FocusOut>", self._on_test_ratio_focus_out)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        top.columnconfigure(3, weight=0)
        top.columnconfigure(4, weight=0)
        top.columnconfigure(5, weight=0)

        middle = ttk.Panedwindow(self.root, orient="horizontal")
        middle.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        left_panel = ttk.Frame(middle, padding=10)
        right_panel = ttk.Frame(middle, padding=8)
        middle.add(left_panel, weight=1)
        middle.add(right_panel, weight=4)

        ttk.Label(left_panel, text="CONTROLS", style="ControlHeading.TLabel").pack(anchor="w", pady=(0, 8))

        models_box = ttk.LabelFrame(left_panel, text="SELECT MODELS", style="Control.TLabelframe", padding=8)
        models_box.pack(fill="both", expand=True, pady=(0, 8))

        self.model_check_frame = tk.Frame(models_box, bg="#f9f9f9", bd=1, relief="solid")
        self.model_check_frame.pack(fill="both", expand=True)

        model_io_box = ttk.LabelFrame(left_panel, text="MODEL MANAGEMENT", style="Control.TLabelframe", padding=8)
        model_io_box.pack(fill="x", pady=(0, 8))
        
        ttk.Button(model_io_box, text="Save Trained Model", command=self._save_model).pack(fill="x", pady=2)
        ttk.Button(model_io_box, text="Load External Model", command=self._load_model).pack(fill="x", pady=2)

        guide_box = ttk.LabelFrame(left_panel, text="QUICK START GUIDE", style="Control.TLabelframe", padding=8)
        guide_box.pack(fill="both", expand=True)
        guide_text_frame = ttk.Frame(guide_box)
        guide_text_frame.pack(fill="both", expand=True)

        self.guide_text = tk.Text(
            guide_text_frame,
            wrap="word",
            relief="solid",
            borderwidth=1,
            bg="#f8f8f8",
            font=("Segoe UI", 9),
        )
        self.guide_text.pack(side="left", fill="both", expand=True)

        guide_scroll = ttk.Scrollbar(guide_text_frame, orient="vertical", command=self.guide_text.yview)
        guide_scroll.pack(side="right", fill="y")
        self.guide_text.configure(yscrollcommand=guide_scroll.set)
        self.guide_text.insert("1.0", QUICK_START_GUIDE_TEXT)
        self.guide_text.configure(state="disabled")

        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        dataset_top10_tab = ttk.Frame(notebook)
        charts_tab = ttk.Frame(notebook)
        predictions_tab = ttk.Frame(notebook)
        combinations_tab = ttk.Frame(notebook)
        table_tab = ttk.Frame(notebook)
        ratio_tab = ttk.Frame(notebook)
        output_tab = ttk.Frame(notebook)

        notebook.add(dataset_top10_tab, text="Dataset Top 10")
        notebook.add(charts_tab, text="Model Comparison")
        notebook.add(predictions_tab, text="Predicted vs Actual")
        notebook.add(combinations_tab, text="Top 10 Combinations")
        notebook.add(table_tab, text="Model Results")
        notebook.add(ratio_tab, text="Ratio Analysis")
        notebook.add(output_tab, text="Output Preview")

        _, dataset_top10_holder, _, dataset_top10_toolbar_holder = self._build_graph_section(
            dataset_top10_tab,
            "Top 10 Best Combinations from Actual Dataset",
            self.open_dataset_top10_fullscreen,
        )

        self.dataset_top10_fig = Figure(figsize=(9, 6), dpi=100)
        self.dataset_top10_canvas = FigureCanvasTkAgg(self.dataset_top10_fig, master=dataset_top10_holder)
        self.dataset_top10_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._enable_canvas_interactions(self.dataset_top10_canvas)
        self._add_canvas_toolbar(dataset_top10_toolbar_holder, self.dataset_top10_canvas)

        _, metrics_holder, self.comparison_filter_frame, metrics_toolbar_holder = self._build_graph_section(
            charts_tab,
            "Model Performance Comparison",
            self.open_metrics_fullscreen,
        )

        self._build_comparison_mode_controls(self.comparison_filter_frame)

        self.metrics_fig = Figure(figsize=(13, 7), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=metrics_holder)
        self.metrics_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._enable_canvas_interactions(self.metrics_canvas)
        self._add_canvas_toolbar(metrics_toolbar_holder, self.metrics_canvas)

        _, predictions_holder, self.prediction_filter_frame, prediction_toolbar_holder = self._build_graph_section(
            predictions_tab,
            "Prediction Charts",
            self.open_predictions_fullscreen,
        )

        # create dedicated sub-frames inside prediction_filter_frame so
        # axis controls and view checkboxes don't clobber each other
        self.prediction_axis_holder = ttk.Frame(self.prediction_filter_frame)
        self.prediction_view_holder = ttk.Frame(self.prediction_filter_frame)

        # Use grid so the axis controls and view checkboxes share space responsively
        self.prediction_axis_holder.grid(row=0, column=0, sticky="nw")
        self.prediction_view_holder.grid(row=0, column=1, sticky="nw", padx=(12, 0))
        try:
            self.prediction_filter_frame.grid_columnconfigure(0, weight=0)
            self.prediction_filter_frame.grid_columnconfigure(1, weight=1)
        except Exception:
            pass

        self._build_prediction_axis_controls(self.prediction_axis_holder)
        # populate view checkboxes into the dedicated view holder
        self._create_view_checkboxes(self.prediction_view_holder, self.prediction_view_vars, self._on_prediction_view_change)

        # bind live-update traces for axis inputs (debounced)
        try:
            self._axis_update_after_id = None
        except Exception:
            self._axis_update_after_id = None
        self._bind_prediction_axis_traces()

        self.pred_fig = Figure(figsize=(9, 6), dpi=100)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=predictions_holder)
        self.pred_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._enable_canvas_interactions(self.pred_canvas)
        self._add_canvas_toolbar(prediction_toolbar_holder, self.pred_canvas)

        _, combinations_holder, self.combination_filter_frame, combination_toolbar_holder = self._build_graph_section(
            combinations_tab,
            "Top 10 Best Combinations by Model",
            self.open_combinations_fullscreen,
        )

        self.combo_fig = Figure(figsize=(9, 6), dpi=100)
        self.combo_canvas = FigureCanvasTkAgg(self.combo_fig, master=combinations_holder)
        self.combo_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._enable_canvas_interactions(self.combo_canvas)
        self._add_canvas_toolbar(combination_toolbar_holder, self.combo_canvas)

        table_frame = ttk.Frame(table_tab, padding=8)
        table_frame.pack(fill="both", expand=True)

        columns = ("Model", "R2", "RMSE", "MSE", "MAE")
        self.results_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        for col in columns:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, anchor="center", width=160)
        self.results_table.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_table.yview)
        y_scroll.pack(side="right", fill="y")
        self.results_table.configure(yscrollcommand=y_scroll.set)

        ratio_table_frame = ttk.Frame(ratio_tab, padding=8)
        ratio_table_frame.pack(fill="x")

        ratio_columns = ("Model", "Ratio", "R2", "RMSE", "MSE", "MAE")
        self.ratio_results_table = ttk.Treeview(ratio_table_frame, columns=ratio_columns, show="headings", height=6)
        for col in ratio_columns:
            self.ratio_results_table.heading(col, text=col)
            self.ratio_results_table.column(col, anchor="center", width=140)
        self.ratio_results_table.pack(side="left", fill="x", expand=True)

        ratio_scroll = ttk.Scrollbar(ratio_table_frame, orient="vertical", command=self.ratio_results_table.yview)
        ratio_scroll.pack(side="right", fill="y")
        self.ratio_results_table.configure(yscrollcommand=ratio_scroll.set)

        _, ratio_holder, ratio_filter_frame, ratio_toolbar_holder = self._build_graph_section(
            ratio_tab,
            "Model Performance vs Train-Test Ratio",
            self.open_ratio_fullscreen,
        )

        self._build_ratio_mode_controls(ratio_filter_frame)

        self.ratio_fig = Figure(figsize=(9, 5.6), dpi=100)
        self.ratio_canvas = FigureCanvasTkAgg(self.ratio_fig, master=ratio_holder)
        self.ratio_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._enable_canvas_interactions(self.ratio_canvas)
        self._add_canvas_toolbar(ratio_toolbar_holder, self.ratio_canvas)

        status_container = ttk.Frame(self.root, padding=(12, 4))
        status_container.pack(fill="x", side="bottom")

        self.progress_bar = ttk.Progressbar(status_container, variable=self.progress_var, maximum=100, style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", pady=(0, 2))

        status = ttk.Label(status_container, textvariable=self.status_text, style="Info.TLabel")
        status.pack(fill="x")

        self._draw_empty_graphs()
        # initialize selectors (store default limits)
        try:
            self._setup_canvas_zoom_selectors(self.dataset_top10_canvas)
            self._setup_canvas_zoom_selectors(self.metrics_canvas)
            self._setup_canvas_zoom_selectors(self.pred_canvas)
            self._setup_canvas_zoom_selectors(self.combo_canvas)
            self._setup_canvas_zoom_selectors(self.ratio_canvas)
        except Exception:
            pass

    def _populate_model_list(self):
        for child in self.model_check_frame.winfo_children():
            child.destroy()

        self.model_select_vars.clear()
        for idx, name in enumerate(MODEL_BUILDERS):
            var = tk.BooleanVar(value=True)
            self.model_select_vars[name] = var
            cb = tk.Checkbutton(
                self.model_check_frame,
                text=name,
                variable=var,
                anchor="w",
                bg="#f9f9f9",
                activebackground="#f9f9f9",
                relief="flat",
                command=self._update_model_notes,
            )
            cb.grid(row=idx, column=0, sticky="w", padx=6, pady=1)

        self.model_check_frame.grid_columnconfigure(0, weight=1)
        self._select_all_models()

    def _draw_empty_graphs(self):
        self._refresh_model_filters()
        self._reset_canvas_hover_state(self.dataset_top10_canvas)
        self._render_dataset_top10_figure(self.dataset_top10_fig, empty=True)
        self._setup_canvas_zoom_selectors(self.dataset_top10_canvas)
        self.dataset_top10_canvas.draw_idle()

        self._reset_canvas_hover_state(self.metrics_canvas)
        self._render_model_comparison_figure(self.metrics_fig, empty=True)
        self._setup_canvas_zoom_selectors(self.metrics_canvas)
        self.metrics_canvas.draw_idle()

        self._reset_canvas_hover_state(self.ratio_canvas)
        self._render_ratio_analysis_figure(self.ratio_fig, empty=True)
        self._setup_canvas_zoom_selectors(self.ratio_canvas)
        self.ratio_canvas.draw_idle()

        self._reset_canvas_hover_state(self.pred_canvas)
        self._render_prediction_figure(self.pred_fig, empty=True)
        self._setup_canvas_zoom_selectors(self.pred_canvas)
        self.pred_canvas.draw_idle()

        self._reset_canvas_hover_state(self.combo_canvas)
        self._render_combination_figure(self.combo_fig, empty=True)
        self._setup_canvas_zoom_selectors(self.combo_canvas)
        self.combo_canvas.draw_idle()

    def _get_best_model_name(self):
        if self.metrics_df.empty:
            return None
        return str(self.metrics_df.iloc[0]["Model"])

    def _refresh_model_filters(self):
        model_names = self.metrics_df["Model"].tolist() if not self.metrics_df.empty else []

        if getattr(self, "prediction_view_holder", None) is not None:
            self._create_view_checkboxes(
                self.prediction_view_holder,
                self.prediction_view_vars,
                self._on_prediction_view_change,
            )

        if self.combination_filter_frame is not None:
            self._create_view_checkboxes(
                self.combination_filter_frame,
                self.combination_view_vars,
                self._on_combination_view_change,
            )

    def _on_prediction_view_change(self, _event=None):
        self._update_prediction_chart()

    def _on_target_col_change(self, _event=None):
        """Update dataset top 10 chart when user changes target column."""
        try:
            if hasattr(self, 'dataset_top10') and self.dataset_top10 is not None:
                df = self._load_dataset(self.dataset_path.get())
                self._compute_dataset_top10(df)
                self._update_dataset_top10_chart()
                self._sync_prediction_units_from_target()
                if self.prediction_frames:
                    self._update_prediction_chart()
                self.status_text.set(f"Dataset Top 10 updated for: {self.target_col_var.get()}")
        except Exception as exc:
            self.status_text.set(f"Error updating target column: {str(exc)}")

    def _on_combination_view_change(self, _event=None):
        self._update_combination_chart()

    def _on_comparison_mode_change(self):
        self._update_metric_charts()

    def _on_ratio_mode_change(self):
        self._update_ratio_chart()

    def _build_metric_bar_colors(self, count, highlight_index, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        if count <= 1:
            return ["#2a9d8f"]

        shades = np.linspace(0.42, 0.88, count)
        colors = [cmap(value) for value in shades]
        colors[highlight_index] = "#2a9d8f"
        return colors

    def _annotate_horizontal_bars(self, ax, values, offset=0.01, fmt="{:.4f}", fontsize=8):
        # All bar value labels disabled - users can hover to see values
        return

    def _annotate_vertical_bars(self, ax, bars, fmt="{:.4f}", fontsize=7):
        for bar in bars:
            height = bar.get_height()
            if abs(height) < 1e-9: continue
            ax.annotate(
                fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=fontsize, alpha=0.9,
                rotation=90
            )
        return

    def _render_model_comparison_figure(self, figure, empty=False):
        figure.clear()

        if empty or self.metrics_df.empty:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "Run models to see the model comparison dashboard", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return

        display_df = self.metrics_df.sort_values("R2", ascending=True).reset_index(drop=True)
        models = display_df["Model"].tolist()
        mse_vals = display_df["MSE"].to_numpy(dtype=float)
        mae_vals = display_df["MAE"].to_numpy(dtype=float)
        rmse_vals = display_df["RMSE"].to_numpy(dtype=float)
        r2_vals = display_df["R2"].to_numpy(dtype=float)

        best_idx = len(display_df) - 1
        best_row = display_df.iloc[best_idx]

        if self.comparison_mode_var.get() == "3D":
            self._render_model_comparison_figure_3d(figure, models, r2_vals, rmse_vals, mse_vals, mae_vals, best_idx, best_row)
            return

        # 2D grouped-column comparison chart (image-2 style)
        wrapped_models = [self._format_model_axis_label(m) for m in models]
        x = np.arange(len(models), dtype=float)
        width = 0.18
        offsets = {
            "R2": -1.5 * width,
            "RMSE": -0.5 * width,
            "MSE": 0.5 * width,
            "MAE": 1.5 * width,
        }

        ax = figure.add_subplot(111)
        bars_r2 = ax.bar(x + offsets["R2"], r2_vals, width=width, color="#3d8ec9", edgecolor="#2b2b2b", linewidth=0.25, label="R2")
        bars_rmse = ax.bar(x + offsets["RMSE"], rmse_vals, width=width, color="#f4a259", edgecolor="#2b2b2b", linewidth=0.25, label="RMSE")
        bars_mse = ax.bar(x + offsets["MSE"], mse_vals, width=width, color="#74c476", edgecolor="#2b2b2b", linewidth=0.25, label="MSE")
        bars_mae = ax.bar(x + offsets["MAE"], mae_vals, width=width, color="#9c89b8", edgecolor="#2b2b2b", linewidth=0.25, label="MAE")

        # Add value labels
        self._annotate_vertical_bars(ax, bars_r2)
        self._annotate_vertical_bars(ax, bars_rmse)
        self._annotate_vertical_bars(ax, bars_mse)
        self._annotate_vertical_bars(ax, bars_mae)

        for model_name, value, bar in zip(models, r2_vals, bars_r2):
            self._set_hover_data(bar, text=f"Model: {model_name}\nR2: {value:.6f}")
        for model_name, value, bar in zip(models, rmse_vals, bars_rmse):
            self._set_hover_data(bar, text=f"Model: {model_name}\nRMSE: {value:.6f}")
        for model_name, value, bar in zip(models, mse_vals, bars_mse):
            self._set_hover_data(bar, text=f"Model: {model_name}\nMSE: {value:.6f}")
        for model_name, value, bar in zip(models, mae_vals, bars_mae):
            self._set_hover_data(bar, text=f"Model: {model_name}\nMAE: {value:.6f}")

        # best model highlight removed for research suitability

        all_vals = np.concatenate([r2_vals, rmse_vals, mse_vals, mae_vals])
        y_top = max(1.0, float(np.max(all_vals)) * 1.35)
        ax.set_ylim(0.0, y_top)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_models, fontsize=9)
        ax.set_title(
            f"Model Metrics Comparison (Grouped) | Best: {best_row['Model']}",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_xlabel("Models", fontweight="bold")
        ax.set_ylabel("Metric Value", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.22)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=8)

        # Apply tight layout to avoid label clipping
        try:
            figure.tight_layout(rect=[0, 0, 0.85, 1])
        except Exception:
            pass

    def _render_model_comparison_figure_3d(self, figure, models, r2_vals, rmse_vals, mse_vals, mae_vals, best_idx, best_row):
        figure.clear()
        outer = figure.add_gridspec(2, 1, height_ratios=[4.9, 1.3], hspace=0.08)
        top = outer[0].subgridspec(2, 2, wspace=0.28, hspace=0.24)
        ax1 = figure.add_subplot(top[0, 0], projection="3d")
        ax2 = figure.add_subplot(top[0, 1], projection="3d")
        ax3 = figure.add_subplot(top[1, 0], projection="3d")
        ax4 = figure.add_subplot(top[1, 1], projection="3d")
        info_ax = figure.add_subplot(outer[1, 0])
        info_ax.set_facecolor("#f6f6f6")
        info_ax.set_xticks([])
        info_ax.set_yticks([])
        for spine in info_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#666666")
            spine.set_linewidth(0.9)

        self._plot_3d_metric_bars(
            ax1,
            models,
            r2_vals * 100.0,
            best_idx,
            "Blues",
            f"Model Accuracy Comparison (3D)\n{best_row['Model']}: {best_row['R2'] * 100:.2f}%",
            "R² Score (%)",
        )

        self._plot_3d_metric_bars(
            ax2,
            models,
            rmse_vals,
            best_idx,
            "Oranges",
            "RMSE Comparison (3D)",
            "RMSE",
        )

        self._plot_3d_metric_bars(
            ax3,
            models,
            mse_vals,
            best_idx,
            "Greens",
            "MSE Comparison (3D)",
            "MSE",
        )

        self._plot_3d_metric_bars(
            ax4,
            models,
            mae_vals,
            best_idx,
            "Purples",
            "MAE Comparison (3D)",
            "MAE",
        )

        self._draw_model_reference_box(info_ax, models, r2_vals * 100.0, rmse_vals, mse_vals, mae_vals)
        figure.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.06)
        # extra tight layout attempt to avoid clipping long labels in 3D
        try:
            figure.tight_layout()
        except Exception:
            pass

    def _draw_model_reference_box(self, ax, models, r2_values, rmse_values, mse_values, mae_values):
        items = []
        for idx, (name, r2_value, rmse_value, mse_value, mae_value) in enumerate(zip(models, r2_values, rmse_values, mse_values, mae_values), 1):
            items.append(f"{idx}. {name} | R2={r2_value:.2f}% | RMSE={rmse_value:.6f} | MSE={mse_value:.6f} | MAE={mae_value:.6f}")

        col_count = 3
        rows = int(math.ceil(len(items) / col_count))
        columns = [items[column_index * rows:(column_index + 1) * rows] for column_index in range(col_count)]
        x_positions = [0.02, 0.35, 0.68]

        ax.text(
            0.015,
            0.97,
            "Model Reference (1=First, 2=Second, etc.)",
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#222222",
            transform=ax.transAxes,
        )

        for column_text, x_pos in zip(columns, x_positions):
            ax.text(
                x_pos,
                0.80,
                "\n".join(column_text),
                ha="left",
                va="top",
                fontsize=7.4,
                family="monospace",
                color="#111111",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.34", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=1.0),
            )

    def _format_model_axis_label(self, name):
        # Use compact multi-line labels so 3D tick text stays readable.
        short_map = {
            "Linear Regression": "Linear\nReg",
            "Gradient Boosting": "Gradient\nBoosting",
            "Random Forest": "Random\nForest",
            "Decision Tree": "Decision\nTree",
            "Extra Trees": "Extra\nTrees",
            "AdaBoost": "Ada\nBoost",
            "LightGBM": "Light\nGBM",
            "XGBoost": "XG\nBoost",
            "Bagging": "Bagging",
            "KNN": "KNN",
            "SVR": "SVR",
        }
        return short_map.get(name, name)

    def _plot_3d_metric_bars(self, ax, labels, values, highlight_index, cmap_name, title, zlabel):
        # Give labels more room by spreading bars a bit further apart.
        x_positions = np.arange(len(labels), dtype=float) * 1.55
        y_positions = np.zeros(len(labels), dtype=float)
        z_positions = np.zeros(len(labels), dtype=float)
        dx = np.full(len(labels), 0.34, dtype=float)
        dy = np.full(len(labels), 0.50, dtype=float)
        colors = self._build_metric_bar_colors(len(labels), highlight_index, cmap_name)

        ax.bar3d(x_positions, y_positions, z_positions, dx, dy, values, color=colors, edgecolor="#2b2b2b", linewidth=0.25)
        # invisible points used for hover on 3D bars
        try:
            hover_scatter = ax.scatter(x_positions + dx / 2, y_positions + dy / 2, values, s=48, c="none", alpha=0.0)
            hover_texts = [f"{label}\n{zlabel}: {value:.6f}" for label, value in zip(labels, values)]
            self._set_hover_data(hover_scatter, texts=hover_texts)
        except Exception:
            pass
        ax.set_title(title, fontweight="bold", fontsize=12, pad=14)
        ax.set_xlabel("Models", labelpad=14, fontweight="bold")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_zlabel(zlabel, labelpad=8)
        ax.set_xticks(x_positions + dx / 2)
        numeric_labels = [str(i + 1) for i in range(len(labels))]
        ax.set_xticklabels(numeric_labels, rotation=0, ha="center", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", pad=1)
        try:
            ax.set_box_aspect((3.5, 1.0, 1.25))
        except Exception:
            pass
        ax.view_init(elev=22, azim=-60)
        try:
            ax.dist = 12.2
        except Exception:
            pass

        max_value = float(np.max(values)) if len(values) else 1.0
        ax.set_zlim(0, max(max_value * 1.15, 1.0))

        ax.grid(False)

    def _render_prediction_figure(self, figure, empty=False):
        figure.clear()

        if empty or self.metrics_df.empty:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "Run models to see Predicted vs Actual plots", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return

        available_names = [name for name in self.metrics_df["Model"].tolist() if name in self.prediction_frames]
        names = self._get_selected_models(self.prediction_view_vars, available_names)
        if not names:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "No prediction data available", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return

        n = len(names)
        cols = 1 if n == 1 else 2
        rows = int(math.ceil(n / cols))
        frame_list = [self.prediction_frames[name] for name in names]
        axis_limits = self._get_prediction_axis_limits(frame_list)
        x_unit = self.prediction_xunit_var.get().strip() or self._infer_prediction_units_from_target(self.dataset_target_col or self.target_col_var.get())
        y_unit = self.prediction_yunit_var.get().strip() or self._infer_prediction_units_from_target(self.dataset_target_col or self.target_col_var.get())

        for idx, name in enumerate(names, start=1):
            ax = figure.add_subplot(rows, cols, idx)
            frame = self.prediction_frames[name]
            # convert to float arrays (no clipping) so original units are preserved
            actual_plot = frame["Actual"].to_numpy(dtype=float)
            pred_plot = frame["Predicted"].to_numpy(dtype=float)
            residual_magnitude = np.abs(actual_plot - pred_plot)

            x_low, x_high, y_low, y_high = axis_limits
            scatter = ax.scatter(
                actual_plot,
                pred_plot,
                s=18,
                alpha=0.9,
                c=residual_magnitude,
                cmap="viridis",
                edgecolors="white",
                linewidths=0.25,
            )
            # attach hover texts for each scatter point using formatted values
            try:
                hover_texts = [
                    f"Model: {name}\nActual: {self._format_feature_value(actual)}\nPredicted: {self._format_feature_value(pred)}\nResidual: {self._format_feature_value(res)}"
                    for actual, pred, res in zip(actual_plot, pred_plot, residual_magnitude)
                ]
                self._set_hover_data(scatter, texts=hover_texts)
            except Exception:
                pass

            # equality line and axis limits based on data
            diag_low = min(x_low, y_low)
            diag_high = max(x_high, y_high)
            ax.plot([diag_low, diag_high], [diag_low, diag_high], "r--", linewidth=1.2)
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)
            metric_row = self.metrics_df[self.metrics_df["Model"] == name].iloc[0]
            ax.set_title(f"{name}", fontweight="bold")

            subplot_label = chr(96 + idx) if idx <= 26 else str(idx)
            ax.text(
                -0.08,
                1.02,
                f"({subplot_label})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#999999", alpha=0.8),
            )

            x_unit_suffix = f" ({x_unit})" if x_unit else ""
            y_unit_suffix = f" ({y_unit})" if y_unit else ""
            ax.set_xlabel(f"Actual{x_unit_suffix}")
            ax.set_ylabel(f"Predicted{y_unit_suffix}")

            box_text = (
                f"X [{x_low:.4f}, {x_high:.4f}]{x_unit_suffix}\n"
                f"Y [{y_low:.4f}, {y_high:.4f}]{y_unit_suffix}\n"
                f"R² = {metric_row['R2']:.4f}\n"
                f"RMSE = {metric_row['RMSE']:.4f}\n"
                f"MSE = {metric_row['MSE']:.4f}\n"
                f"MAE = {metric_row['MAE']:.4f}"
            )
            ax.text(
                0.03,
                0.86,
                box_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.3,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#bbbbbb", alpha=0.85),
            )

            # colorbar for residual magnitude (auto-scaled)
            figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="|Residual|")

        figure.tight_layout()

    def _format_feature_value(self, value):
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            if value == 0:
                return "0"
            abs_val = abs(value)
            if abs_val >= 1e9:
                return f"{value / 1e9:.2f} G"
            elif abs_val >= 1e6:
                return f"{value / 1e6:.2f} M"
            elif abs_val >= 1e3:
                return f"{value / 1e3:.2f} k"
            elif abs_val >= 1.0:
                return f"{value:.4f}".rstrip("0").rstrip(".")
            elif abs_val >= 1e-3:
                return f"{value * 1e3:.2f} milli"
            elif abs_val >= 1e-6:
                return f"{value * 1e6:.2f} micro"
            elif abs_val >= 1e-9:
                return f"{value * 1e9:.2f} nano"
            elif abs_val >= 1e-12:
                return f"{value * 1e12:.2f} pico"
            else:
                return f"{value:.2e}"
        return str(value)

    def _format_combination_name(self, row, feature_cols):
        parts = [f"{col}={self._format_feature_value(row[col])}" for col in feature_cols]
        return " | ".join(parts)

    def _build_top10_details_text(self, top_df, wrap_width=78):
        feature_cols = [c for c in top_df.columns if c not in ("Combination", "Predicted")]
        lines = ["Top-10 Combination Details"]

        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            params = ", ".join(
                f"{col.split('(')[0].strip()}={self._format_feature_value(row[col])}" for col in feature_cols
            )
            line = f"C{rank}: {params} | y={row['Predicted']:.4f}"
            lines.append(textwrap.fill(line, width=wrap_width, subsequent_indent="    "))

        return "\n".join(lines)

    def _render_combination_figure(self, figure, empty=False):
        figure.clear()

        if empty or not self.top_combinations:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "Run models to see top 10 best combinations", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return

        available_names = [name for name in self.metrics_df["Model"].tolist() if name in self.top_combinations]
        model_names = self._get_selected_models(self.combination_view_vars, available_names)
        if not model_names:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "No combination data available", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return

        if len(model_names) == 1:
            model_name = model_names[0]
            top_df = self.top_combinations[model_name].sort_values("Predicted", ascending=False).reset_index(drop=True)
            feature_cols = [c for c in top_df.columns if c not in ("Combination", "Predicted")]

            grid = figure.add_gridspec(1, 2, width_ratios=[1.40, 1.30], wspace=0.04)
            ax_bar = figure.add_subplot(grid[0, 0])
            ax_text = figure.add_subplot(grid[0, 1])

            labels = [f"C{i + 1}" for i in range(len(top_df))]
            values = top_df["Predicted"].to_numpy(dtype=float)
            colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(values)))

            bars = ax_bar.bar(labels, values, color=colors, edgecolor="none")
            for rank, (bar, (_, row)) in enumerate(zip(bars, top_df.iterrows()), start=1):
                details = [f"{col}: {self._format_feature_value(row[col])}" for col in feature_cols]
                tooltip = f"Rank: C{rank}\nPredicted: {row['Predicted']:.6f}\n" + "\n".join(details)
                self._set_hover_data(bar, text=tooltip)
            # value labels removed per user request
            ax_bar.set_title(f"{model_name} - Top 10 Best Combinations", fontsize=12, fontweight="bold")
            ax_bar.set_xlabel("Combination Rank")
            ax_bar.set_ylabel("Predicted Target")
            ax_bar.tick_params(axis="x", rotation=0)

            ymin = float(np.min(values)) if len(values) else 0.0
            ymax = float(np.max(values)) if len(values) else 1.0
            if np.isclose(ymin, ymax):
                pad = 0.02 if np.isclose(ymax, 0.0) else abs(ymax) * 0.03
                ax_bar.set_ylim(ymin - pad, ymax + pad)
            else:
                pad = (ymax - ymin) * 0.08
                ax_bar.set_ylim(ymin - pad, ymax + pad)

            # value labels on bars removed per user request

            wrap_width = 86 if figure.get_figwidth() >= 15 else 74
            details_text = self._build_top10_details_text(top_df, wrap_width=wrap_width)
            line_count = details_text.count("\n") + 1
            details_font_size = 8.8 if line_count <= 24 else 8.0

            ax_text.axis("off")
            ax_text.text(
                0.02,
                0.50,
                details_text,
                va="center",
                ha="left",
                fontsize=details_font_size,
                transform=ax_text.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#444444", alpha=1.0),
            )

            figure.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12)
            return

        n = len(model_names)
        cols = 1 if n == 1 else 2
        rows = int(math.ceil(n / cols))

        for idx, model_name in enumerate(model_names, start=1):
            ax = figure.add_subplot(rows, cols, idx)
            top_df = self.top_combinations[model_name].sort_values("Predicted", ascending=False).reset_index(drop=True)
            ranks = [f"Top {rank}" for rank in range(1, len(top_df) + 1)]
            values = top_df["Predicted"].to_numpy(dtype=float)[::-1]
            bars = ax.barh(ranks[::-1], values, color="#2a9d8f")
            ranked_rows = top_df.iloc[::-1].reset_index(drop=True)
            for rank_name, bar, (_, row) in zip(ranks[::-1], bars, ranked_rows.iterrows()):
                self._set_hover_data(bar, text=f"{rank_name}\nPredicted: {row['Predicted']:.6f}")
            # value labels removed per user request
            best_value = float(top_df["Predicted"].iloc[0])
            ax.set_title(f"{model_name} - Top 10 (Best={best_value:.4f})")
            ax.set_xlabel("Predicted Target")
            ax.tick_params(axis="y", labelsize=9)

        figure.tight_layout()

    def _open_fullscreen_window(self, title, render_function):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.attributes("-fullscreen", True)
        window.configure(background="#111111")

        def close_window(_event=None):
            window.destroy()

        window.bind("<Escape>", close_window)

        wrapper = ttk.Frame(window, padding=10)
        wrapper.pack(fill="both", expand=True)

        topbar = ttk.Frame(wrapper)
        topbar.pack(fill="x", pady=(0, 8))

        ttk.Label(topbar, text=title, style="Title.TLabel").pack(side="left")
        ttk.Button(topbar, text="Close", command=close_window).pack(side="right")

        fig = Figure(figsize=(22, 11), dpi=110)
        canvas = FigureCanvasTkAgg(fig, master=wrapper)
        self._enable_canvas_interactions(canvas)

        toolbar_frame = ttk.Frame(wrapper)
        toolbar_frame.pack(fill="x", pady=(0, 6))

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left")

        ttk.Button(toolbar_frame, text="Reset Zoom", command=lambda: self._reset_zoom(canvas)).pack(side="left", padx=(8, 0))
        ttk.Label(toolbar_frame, text="Left drag: zoom | Right click: reset").pack(side="left", padx=(12, 0))

        canvas.get_tk_widget().pack(fill="both", expand=True)

        def redraw_fullscreen():
            self._reset_canvas_hover_state(canvas)
            render_function(fig)
            self._setup_canvas_zoom_selectors(canvas)
            canvas.draw_idle()

        redraw_fullscreen()

        def on_resize(event):
            if event.width < 200 or event.height < 150:
                return
            fig.set_size_inches(event.width / fig.dpi, event.height / fig.dpi, forward=True)
            redraw_fullscreen()

        wrapper.bind("<Configure>", on_resize)

    def open_metrics_fullscreen(self):
        self._open_fullscreen_window("Model Comparison Full Screen", self._render_model_comparison_figure)

    def open_dataset_top10_fullscreen(self):
        self._open_fullscreen_window("Dataset Top 10 Full Screen", self._render_dataset_top10_figure)

    def open_predictions_fullscreen(self):
        self._open_fullscreen_window("Prediction Charts Full Screen", self._render_prediction_figure)

    def open_combinations_fullscreen(self):
        self._open_fullscreen_window("Top 10 Combinations Full Screen", self._render_combination_figure)

    def open_ratio_fullscreen(self):
        self._open_fullscreen_window("Ratio Analysis Full Screen", self._render_ratio_analysis_figure)

    def _browse_dataset(self):
        selected = filedialog.askopenfilename(
            title="Choose CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.dataset_path.set(selected)
            self._show_dataset_info()

    def _get_selected_model_names(self):
        return [name for name, var in self.model_select_vars.items() if var.get()]

    def _select_all_models(self):
        for var in self.model_select_vars.values():
            var.set(True)
        self._update_model_notes()

    def _clear_selection(self):
        for var in self.model_select_vars.values():
            var.set(False)
        self._update_model_notes()

    def _update_model_notes(self, _event=None):
        names = self._get_selected_model_names()
        if not names:
            self.status_text.set("No model selected.")
            return

        self.status_text.set(f"Selected models: {', '.join(names)}")

    def _show_dataset_info(self):
        try:
            dataset = self._load_dataset(self.dataset_path.get())
        except Exception as exc:
            messagebox.showerror("Dataset Error", str(exc))
            self.status_text.set("Dataset loading failed. Please check the file and try again.")
            return

        # Populate target column dropdown with all column names
        all_cols = dataset.columns.tolist()
        self.target_col_combo['values'] = all_cols
        
        # Set default target to last column
        default_target = dataset.columns[-1]
        self.target_col_var.set(default_target)
        
        # Get features excluding the selected target
        target_col = default_target
        feature_cols = [col for col in all_cols if col != target_col]

        msg = (
            f"Rows: {len(dataset)}\n"
            f"Columns: {len(dataset.columns)}\n"
            f"All columns: {', '.join(all_cols)}\n"
            f"Target column (you can change): {target_col}\n"
            f"Feature columns: {', '.join(feature_cols)}"
        )
        messagebox.showinfo("Dataset Information", msg)
        self._compute_dataset_top10(dataset)
        self._update_dataset_top10_chart()
        self._sync_prediction_units_from_target()
        if self.prediction_frames:
            self._update_prediction_chart()
        self.notebook.select(0)
        self.status_text.set("Dataset loaded successfully. Select target column if needed, then click 'Run Selected Models'.")

    def _load_dataset(self, path_str: str) -> pd.DataFrame:
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

    def _get_features_and_target(self, df: pd.DataFrame):
        """
        Extract features (X) and target (y) based on selected target column.
        Returns: (X, y, target_col_name, feature_cols_list)
        """
        target_col = self.target_col_var.get()
        if not target_col or target_col not in df.columns:
            # Fallback to last column if not selected
            target_col = df.columns[-1]
            self.target_col_var.set(target_col)
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y, target_col, feature_cols

    def _compute_dataset_top10(self, df: pd.DataFrame):
        _, _, target_col, feature_cols = self._get_features_and_target(df)
        top10 = df.nlargest(10, target_col).reset_index(drop=True)
        top10_display = top10.copy()
        top10_display["Combination"] = top10_display.apply(
            lambda row: self._format_combination_name(row, feature_cols),
            axis=1,
        )
        top10_display = top10_display.rename(columns={target_col: "Actual"})
        self.dataset_top10 = top10_display
        self.dataset_target_col = target_col

    def _build_dataset_top10_details_text(self, top_df, feature_cols, wrap_width=78):
        lines = ["Top-10 Dataset Combination Details"]
        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            params = ", ".join(
                f"{col.split('(')[0].strip()}={self._format_feature_value(row[col])}" for col in feature_cols
            )
            line = f"C{rank}: {params} | {self.dataset_target_col}={row['Actual']:.4f}"
            lines.append(textwrap.fill(line, width=wrap_width, subsequent_indent="    "))
        return "\n".join(lines)

    def _render_dataset_top10_figure(self, figure, empty=False):
        figure.clear()

        if empty or self.dataset_top10 is None:
            ax = figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Load a dataset to see the top 10 best combinations\nfrom the actual data",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.axis("off")
            return

        top_df = self.dataset_top10.copy()
        feature_cols = [c for c in top_df.columns if c not in ("Combination", "Actual")]

        grid = figure.add_gridspec(1, 2, width_ratios=[1.40, 1.30], wspace=0.04)
        ax_bar = figure.add_subplot(grid[0, 0])
        ax_text = figure.add_subplot(grid[0, 1])

        labels = [f"C{i + 1}" for i in range(len(top_df))]
        values = top_df["Actual"].to_numpy(dtype=float)
        colors = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, len(values)))

        bars = ax_bar.bar(labels, values, color=colors, edgecolor="none")
        for rank, (bar, (_, row)) in enumerate(zip(bars, top_df.iterrows()), start=1):
            details = [f"{col}: {self._format_feature_value(row[col])}" for col in feature_cols]
            tooltip = f"Rank: C{rank}\nActual: {row['Actual']:.6f}\n" + "\n".join(details)
            self._set_hover_data(bar, text=tooltip)
        self._annotate_vertical_bars(ax_bar, bars, fmt="{:.4f}", fontsize=9)
        ax_bar.set_title(f"Dataset Top 10 - Best {self.dataset_target_col} Values", fontsize=12, fontweight="bold")
        ax_bar.set_xlabel("Combination Rank")
        ax_bar.set_ylabel(f"Actual {self.dataset_target_col}")
        ax_bar.tick_params(axis="x", rotation=0)

        ymin = float(np.min(values)) if len(values) else 0.0
        ymax = float(np.max(values)) if len(values) else 1.0
        if np.isclose(ymin, ymax):
            pad = 0.02 if np.isclose(ymax, 0.0) else abs(ymax) * 0.03
            ax_bar.set_ylim(ymin - pad, ymax + pad)
        else:
            pad = (ymax - ymin) * 0.08
            ax_bar.set_ylim(ymin - pad, ymax + pad)

        wrap_width = 86 if figure.get_figwidth() >= 15 else 74
        details_text = self._build_dataset_top10_details_text(top_df, feature_cols, wrap_width=wrap_width)
        line_count = details_text.count("\n") + 1
        details_font_size = 8.8 if line_count <= 24 else 8.0

        ax_text.axis("off")
        ax_text.text(
            0.02,
            0.50,
            details_text,
            va="center",
            ha="left",
            fontsize=details_font_size,
            transform=ax_text.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1", edgecolor="#FF8F00", alpha=1.0),
        )

        try:
            figure.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12)
        except Exception:
            pass

    def _update_dataset_top10_chart(self):
        self._reset_canvas_hover_state(self.dataset_top10_canvas)
        self._render_dataset_top10_figure(self.dataset_top10_fig)
        self._setup_canvas_zoom_selectors(self.dataset_top10_canvas)
        self.dataset_top10_canvas.draw_idle()

    def _parse_percent_value(self, raw_value: str):
        value_text = str(raw_value).strip()
        if value_text == "":
            return None

        percent = float(value_text)
        if percent < 0 or percent > 100:
            raise ValueError("Ratio must be between 0 and 100.")
        return percent

    def _format_percent_text(self, percent: float):
        return f"{percent:.2f}".rstrip("0").rstrip(".")

    def _set_split_from_train_percent(self, train_percent: float):
        train_size = train_percent / 100.0
        if train_size <= 0.0 or train_size >= 1.0:
            raise ValueError("Train ratio must be greater than 0 and less than 100 for train/test split.")

        self.train_size_value = train_size
        self.test_size_value = 1.0 - train_size

    def _sync_ratio_pair(self, source="train"):
        if self._ratio_sync_in_progress:
            return

        self._ratio_sync_in_progress = True
        try:
            if source == "train":
                train_percent = self._parse_percent_value(self.train_ratio_var.get())
                if train_percent is None:
                    return
                test_percent = 100.0 - train_percent
                self.test_ratio_var.set(self._format_percent_text(test_percent))
                self._set_split_from_train_percent(train_percent)
            else:
                test_percent = self._parse_percent_value(self.test_ratio_var.get())
                if test_percent is None:
                    return
                train_percent = 100.0 - test_percent
                self.train_ratio_var.set(self._format_percent_text(train_percent))
                self._set_split_from_train_percent(train_percent)
        finally:
            self._ratio_sync_in_progress = False

    def _on_train_ratio_change(self, _event=None):
        try:
            self._sync_ratio_pair(source="train")
        except ValueError:
            pass

    def _on_test_ratio_change(self, _event=None):
        try:
            self._sync_ratio_pair(source="test")
        except ValueError:
            pass

    def _on_train_ratio_focus_out(self, _event=None):
        try:
            self._sync_ratio_pair(source="train")
        except ValueError as exc:
            messagebox.showerror("Invalid Ratio", str(exc))
            self.train_ratio_var.set("80")
            self.test_ratio_var.set("20")
            self._set_split_from_train_percent(80.0)

    def _on_test_ratio_focus_out(self, _event=None):
        try:
            self._sync_ratio_pair(source="test")
        except ValueError as exc:
            messagebox.showerror("Invalid Ratio", str(exc))
            self.train_ratio_var.set("80")
            self.test_ratio_var.set("20")
            self._set_split_from_train_percent(80.0)

    def _parse_train_ratio_list(self):
        default_ratios = [0.8, 0.7, 0.6]
        text = self.ratio_list_var.get().strip()

        ratios = list(default_ratios)
        for token in [part.strip() for part in text.split(",") if part.strip()]:
            numeric = float(token)
            percent = numeric * 100.0 if numeric <= 1.0 else numeric
            if percent <= 0 or percent >= 100:
                raise ValueError("Each train ratio in the list must be greater than 0 and less than 100.")
            ratios.append(round(percent / 100.0, 4))

        if not ratios:
            raise ValueError("Please provide at least one train ratio.")

        deduped = list(dict.fromkeys(ratios))
        return deduped

    def _ratio_label(self, train_size: float):
        train_pct = int(round(train_size * 100))
        test_pct = 100 - train_pct
        return f"{train_pct}-{test_pct}"

    def _build_model_pipeline(self, model_name: str):
        model = MODEL_BUILDERS[model_name]()
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    def _compute_regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(mae),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def _train_single_model(self, model_name, X_train, X_test, y_train, y_test):
        pipeline = self._build_model_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = self._compute_regression_metrics(y_test, y_pred)
        pred_df = pd.DataFrame({
            "Actual": y_test.reset_index(drop=True),
            "Predicted": pd.Series(y_pred),
            "Residual": y_test.reset_index(drop=True) - pd.Series(y_pred),
        })
        
        # store trained model for saving later
        self.trained_models[model_name] = pipeline
        
        return pipeline, metrics, pred_df

    def _save_model(self):
        """
        Allows users to save selected trained models into .pkl files using joblib.
        """
        if not self.trained_models:
            messagebox.showwarning("No Models", "No models have been trained yet. Please run models first.")
            return

        # Let user choose which model to save if multiple exist
        model_names = list(self.trained_models.keys())
        if len(model_names) > 1:
            # Create an improved selection dialog
            select_win = tk.Toplevel(self.root)
            select_win.title("Select Model(s) to Save")
            select_win.geometry("380x480")
            select_win.transient(self.root)
            select_win.grab_set()

            ttk.Label(select_win, text="Select models to export:", font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))
            ttk.Label(select_win, text="Tip: Use Ctrl+A or Click + Drag to select multiple", font=("Segoe UI", 8, "italic")).pack(pady=(0, 10))
            
            lb = tk.Listbox(select_win, font=("Segoe UI", 10), selectmode="extended", borderwidth=1, relief="solid")
            lb.pack(fill="both", expand=True, padx=15, pady=5)
            for m in model_names:
                lb.insert("end", m)
            
            # Pre-select all to make it easier for user
            lb.selection_set(0, "end")

            def select_all_action(event=None):
                lb.selection_set(0, "end")
                return "break"

            # Bind Ctrl+A
            lb.bind("<Control-a>", select_all_action)
            lb.bind("<Control-A>", select_all_action)

            btn_frame = ttk.Frame(select_win)
            btn_frame.pack(fill="x", padx=15, pady=20)

            def do_save():
                indices = lb.curselection()
                if not indices:
                    messagebox.showwarning("No Selection", "Please select at least one model to save.")
                    return
                
                selected = [lb.get(i) for i in indices]
                select_win.destroy()
                
                if len(selected) == 1:
                    # Single file save
                    self._execute_save_dump(selected[0])
                else:
                    # Batch folder save
                    self._execute_batch_save(selected)

            ttk.Button(btn_frame, text="Select All (Ctrl+A)", command=select_all_action).pack(side="left", expand=True, fill="x", padx=2)
            ttk.Button(btn_frame, text="Save Selected", command=do_save, style="Accent.TButton").pack(side="left", expand=True, fill="x", padx=2)
        else:
            self._execute_save_dump(model_names[0])

    def _execute_batch_save(self, model_names):
        """
        Saves multiple models into a selected directory.
        """
        try:
            target_dir = filedialog.askdirectory(title="Select Folder to Save Multiple Models")
            if not target_dir:
                return
            
            target_path = Path(target_dir)
            saved_count = 0
            for name in model_names:
                model = self.trained_models[name]
                file_name = f"{name.replace(' ', '_')}_model.pkl"
                # saving each model using joblib
                joblib.dump(model, target_path / file_name)
                saved_count += 1
            
            self.status_text.set(f"Successfully saved {saved_count} models to folder: {target_path.name}")
            messagebox.showinfo("Batch Save Success", f"Successfully saved {saved_count} models to:\n{target_dir}")
        except Exception as e:
            self.status_text.set(f"Batch save failed: {str(e)}")
            messagebox.showerror("Save Error", f"Could not save models:\n{str(e)}")

    def _execute_save_dump(self, model_name):
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                initialfile=f"{model_name.replace(' ', '_')}_model.pkl"
            )
            if not path:
                return

            model = self.trained_models[model_name]
            # saving model using joblib
            joblib.dump(model, path)
            
            self.status_text.set(f"Successfully saved {model_name} to {Path(path).name}")
            messagebox.showinfo("Success", f"Model '{model_name}' saved successfully!")
        except Exception as e:
            self.status_text.set(f"Failed to save model: {str(e)}")
            messagebox.showerror("Save Error", f"Could not save model:\n{str(e)}")

    def _load_model(self):
        """
        Loads a .pkl model file using joblib and integrates it into the application memory.
        """
        try:
            path = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if not path:
                return

            # loading model using joblib
            loaded_pipeline = joblib.load(path)
            
            # Check if it's a valid sklearn pipeline or model (basic check)
            if not hasattr(loaded_pipeline, "predict"):
                raise ValueError("The selected file does not appear to be a valid machine learning model.")

            model_name = f"Loaded_{Path(path).stem}"
            self.trained_models[model_name] = loaded_pipeline
            
            self.status_text.set(f"Successfully loaded model: {model_name}")
            
            # If a dataset is currently loaded, allow prediction without retraining
            if not self.dataset_path.get() or not Path(self.dataset_path.get()).exists():
                messagebox.showinfo("Model Loaded", f"Model '{model_name}' loaded successfully.\n\nTo see results, please load a dataset.")
                return

            # Try to integrate loaded model with existing prediction workflow
            self._run_inference_on_loaded_model(model_name, loaded_pipeline)
            
            messagebox.showinfo("Success", f"Model '{model_name}' loaded and integrated with current data!")
            
        except Exception as e:
            self.status_text.set(f"Failed to load model: {str(e)}")
            messagebox.showerror("Load Error", f"Invalid or corrupted model file:\n{str(e)}")

    def _run_inference_on_loaded_model(self, name, pipeline):
        """
        Prediction using loaded model on current dataset.
        """
        try:
            df = self._load_dataset(self.dataset_path.get())
            X, y, target_col, feature_cols = self._get_features_and_target(df)
            
            # We use the current test split if possible, or just the whole set
            # For consistency with the GUI workflow, we'll do a fresh split with RANDOM_STATE
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(self.test_size_value), random_state=RANDOM_STATE
            )

            y_pred = pipeline.predict(X_test)
            metrics = self._compute_regression_metrics(y_test, y_pred)
            
            pred_df = pd.DataFrame({
                "Actual": y_test.reset_index(drop=True),
                "Predicted": pd.Series(y_pred),
                "Residual": y_test.reset_index(drop=True) - pd.Series(y_pred),
            })

            # Update application state
            if self.metrics_df.empty:
                self.metrics_df = pd.DataFrame([{
                    "Model": name, "R2": metrics["R2"], "RMSE": metrics["RMSE"], "MSE": metrics["MSE"], "MAE": metrics["MAE"]
                }])
            else:
                # Remove if already exists with same name
                self.metrics_df = self.metrics_df[self.metrics_df["Model"] != name]
                new_row = pd.DataFrame([{
                    "Model": name, "R2": metrics["R2"], "RMSE": metrics["RMSE"], "MSE": metrics["MSE"], "MAE": metrics["MAE"]
                }])
                self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True).sort_values("R2", ascending=False)
            
            self.prediction_frames[name] = pred_df
            
            # Update GUI
            self._refresh_model_filters()
            self._update_results_table()
            self._update_metric_charts()
            self._update_prediction_chart()
            
        except Exception as e:
            messagebox.showwarning("Inference Warning", f"Model loaded but could not run on current data:\n{str(e)}\n\nEnsure features match.")

    def _run_ratio_analysis(self, X, y, model_names, train_sizes):
        rows = []

        for model_name in model_names:
            for train_size in train_sizes:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=(1.0 - train_size),
                    random_state=RANDOM_STATE,
                )
                _, metrics, _ = self._train_single_model(model_name, X_train, X_test, y_train, y_test)
                ratio_text = self._ratio_label(train_size)
                rows.append({
                    "Model": model_name,
                    "Ratio": ratio_text,
                    "TrainSize": train_size,
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MSE": metrics["MSE"],
                    "MAE": metrics["MAE"],
                })

        ratio_df = pd.DataFrame(rows)
        return ratio_df

    def _render_ratio_analysis_figure(self, figure, empty=False):
        figure.clear()

        if empty or self.ratio_metrics_df.empty:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "Run models to see ratio-based R2 and RMSE analysis", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return

        selected_model = self.ratio_model_var.get()
        if not selected_model or selected_model not in self.ratio_metrics_df["Model"].unique():
            selected_model = self.ratio_metrics_df["Model"].unique()[0]
            self.ratio_model_var.set(selected_model)

        plot_df = self.ratio_metrics_df[self.ratio_metrics_df["Model"] == selected_model].copy()
        
        ratio_labels = plot_df["Ratio"].tolist()
        r2_vals = plot_df["R2"].to_numpy(dtype=float)
        rmse_vals = plot_df["RMSE"].to_numpy(dtype=float)
        mse_vals = plot_df["MSE"].to_numpy(dtype=float)
        mae_vals = plot_df["MAE"].to_numpy(dtype=float)

        if self.ratio_mode_var.get() == "2D":
            x = np.arange(len(ratio_labels), dtype=float)
            metrics = [
                ("R2", r2_vals, "#3d8ec9"),
                ("RMSE", rmse_vals, "#f4a259"),
                ("MSE", mse_vals, "#74c476"),
                ("MAE", mae_vals, "#9c89b8"),
            ]
            width = 0.18
            
            ax = figure.add_subplot(111)
            # best ratio highlight removed for research suitability
            all_vals = np.concatenate([r2_vals, rmse_vals, mse_vals, mae_vals])
            y_top = max(float(np.max(all_vals)) * 1.35, 1.0)
            ax.set_ylim(0.0, y_top)
            for idx, (metric_name, metric_vals, metric_color) in enumerate(metrics):
                offset = (idx - 1.5) * width
                bars = ax.bar(x + offset, metric_vals, width=width, color=metric_color, label=metric_name, edgecolor="#2b2b2b", linewidth=0.25)
                self._annotate_vertical_bars(ax, bars)
                self._set_hover_data(
                    bars,
                    texts=[f"Ratio: {label}\n{metric_name}: {value:.6f}" for label, value in zip(ratio_labels, metric_vals)],
                )

            ax.set_title(f"Ratio Analysis - {selected_model}", fontweight="bold", pad=10)
            ax.set_xlabel("Train-Test Ratio")
            ax.set_ylabel("Metric Value")

            ax.set_xticks(x)
            ax.set_xticklabels(ratio_labels)
            y_top = max(float(np.max(all_vals)) * 1.2, 1.0)
            ax.set_ylim(0.0, y_top)

            ax.grid(axis="both", linestyle="--", alpha=0.25)
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

            figure.tight_layout(rect=[0, 0, 0.85, 1])
            return

        ax = figure.add_subplot(111, projection="3d")

        x = np.arange(len(ratio_labels), dtype=float)
        dx = np.full(len(ratio_labels), 0.38, dtype=float)
        dy = np.full(len(ratio_labels), 0.35, dtype=float)
        metric_specs = [
            ("R2", r2_vals, "#2a9d8f", 0.0),
            ("RMSE", rmse_vals, "#e76f51", 1.0),
            ("MSE", mse_vals, "#3a86ff", 2.0),
            ("MAE", mae_vals, "#8338ec", 3.0),
        ]

        for metric_name, metric_vals, metric_color, y_base in metric_specs:
            ax.bar3d(
                x,
                np.full(len(x), y_base),
                np.zeros(len(x)),
                dx,
                dy,
                metric_vals,
                color=metric_color,
                edgecolor="#222222",
                alpha=0.92,
            )
        try:
            for metric_name, metric_vals, _metric_color, y_base in metric_specs:
                hover = ax.scatter(
                    x + dx / 2,
                    np.full(len(x), y_base) + dy / 2,
                    metric_vals,
                    s=40,
                    c="none",
                    alpha=0.0,
                )
                self._set_hover_data(
                    hover,
                    texts=[f"Ratio: {label}\n{metric_name}: {value:.6f}" for label, value in zip(ratio_labels, metric_vals)],
                )
        except Exception:
            pass

        ax.set_title(f"Ratio Analysis - {selected_model}", fontweight="bold", pad=16)
        ax.set_xlabel("Train-Test Ratio", labelpad=10)
        ax.set_ylabel("Metrics", labelpad=10)
        ax.set_zlabel("Values", labelpad=10)

        ax.set_xticks(x + dx / 2)
        ax.set_xticklabels(ratio_labels)
        ax.set_yticks([0 + dy[0] / 2, 1 + dy[0] / 2, 2 + dy[0] / 2, 3 + dy[0] / 2])
        ax.set_yticklabels(["R2", "RMSE", "MSE", "MAE"])
        ax.view_init(elev=23, azim=-58)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="#2a9d8f", edgecolor="#222222", label="R2"),
            Patch(facecolor="#e76f51", edgecolor="#222222", label="RMSE"),
            Patch(facecolor="#3a86ff", edgecolor="#222222", label="MSE"),
            Patch(facecolor="#8338ec", edgecolor="#222222", label="MAE"),
        ]
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.05, 0.5))

        z_max = max(float(np.max(r2_vals)), float(np.max(rmse_vals)), float(np.max(mse_vals)), float(np.max(mae_vals)))
        ax.set_zlim(0, max(z_max * 1.15, 1.0))
        figure.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.08)

    def _update_ratio_results_table(self):
        for item in self.ratio_results_table.get_children():
            self.ratio_results_table.delete(item)

        for _, row in self.ratio_metrics_df.iterrows():
            self.ratio_results_table.insert(
                "",
                "end",
                values=(
                    row["Model"],
                    row["Ratio"],
                    f"{row['R2']:.6f}",
                    f"{row['RMSE']:.6f}",
                    f"{row['MSE']:.6f}",
                    f"{row['MAE']:.6f}",
                ),
            )

    def run_selected_models(self):
        global LIGHTGBM_FALLBACK_MESSAGE, XGBOOST_FALLBACK_MESSAGE
        selected_models = self._get_selected_model_names()
        if not selected_models:
            messagebox.showwarning("No Model Selected", "Please select at least one model.")
            return

        self.progress_var.set(5)
        self.root.update_idletasks()

        LIGHTGBM_FALLBACK_MESSAGE = None
        XGBOOST_FALLBACK_MESSAGE = None

        try:
            self._sync_ratio_pair(source="train")
            requested_train_sizes = self._parse_train_ratio_list()
            if round(self.train_size_value, 4) not in requested_train_sizes:
                requested_train_sizes.insert(0, round(self.train_size_value, 4))

            self.progress_var.set(15)
            df = self._load_dataset(self.dataset_path.get())
            X, y, target_col, feature_cols = self._get_features_and_target(df)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=float(self.test_size_value),
                random_state=RANDOM_STATE,
            )
            self.progress_var.set(25)

            records = []
            predictions = {}
            top_combinations = {}

            total_models = len(selected_models)
            for i, model_name in enumerate(selected_models):
                self.status_text.set(f"Training {model_name}...")
                self.root.update_idletasks()
                
                pipeline, metrics, pred_df = self._train_single_model(model_name, X_train, X_test, y_train, y_test)

                records.append({
                    "Model": model_name,
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MSE": metrics["MSE"],
                    "MAE": metrics["MAE"],
                })
                predictions[model_name] = pred_df

                combo_features = X.drop_duplicates().reset_index(drop=True)
                combo_preds = pipeline.predict(combo_features)
                combo_df = combo_features.copy()
                combo_df["Predicted"] = combo_preds
                combo_df = combo_df.sort_values("Predicted", ascending=False).head(10).reset_index(drop=True)
                combo_df["Combination"] = combo_df.apply(
                    lambda row: self._format_combination_name(row, feature_cols),
                    axis=1,
                )
                top_combinations[model_name] = combo_df[["Combination", "Predicted"] + feature_cols]
                
                # update progress
                prog = 25 + (i + 1) / total_models * 60
                self.progress_var.set(prog)

            self.metrics_df = pd.DataFrame(records).sort_values("R2", ascending=False).reset_index(drop=True)
            self.prediction_frames = predictions
            self.top_combinations = top_combinations

            self.status_text.set("Running ratio analysis...")
            self.root.update_idletasks()
            ratio_df = self._run_ratio_analysis(X, y, selected_models, requested_train_sizes)
            self.ratio_metrics_df = ratio_df
            self.ratio_best_r2_row = ratio_df.loc[ratio_df["R2"].idxmax()]
            self.ratio_best_rmse_row = ratio_df.loc[ratio_df["RMSE"].idxmin()]
            self.ratio_best_mse_row = ratio_df.loc[ratio_df["MSE"].idxmin()]
            self.ratio_best_mae_row = ratio_df.loc[ratio_df["MAE"].idxmin()]

            # update ratio model selector values
            if hasattr(self, 'ratio_model_combo'):
                self.ratio_model_combo['values'] = list(ratio_df["Model"].unique())
                if not self.ratio_model_var.get() or self.ratio_model_var.get() not in self.ratio_model_combo['values']:
                    self.ratio_model_var.set(str(ratio_df.iloc[0]["Model"]))

            self.progress_var.set(95)
            self.status_text.set("Updating UI...")
            self._refresh_model_filters()
            self._update_results_table()
            self._update_ratio_results_table()
            self._update_metric_charts()
            self._update_ratio_chart()
            
            self.progress_var.set(100)
            self.status_text.set("All models trained and results updated.")
            messagebox.showinfo("Success", "All selected models have been trained successfully.")
            self._update_prediction_chart()
            self._update_combination_chart()
            self._update_output_preview(df)

            best_model = self.metrics_df.iloc[0]
            self.status_text.set(
                f"Run complete. Best model: {best_model['Model']} (R²={best_model['R2']:.4f}, RMSE={best_model['RMSE']:.4f}, MSE={best_model['MSE']:.4f}, MAE={best_model['MAE']:.4f}) | "
                f"Best ratio by R²: {self.ratio_best_r2_row['Ratio']} ({self.ratio_best_r2_row['R2']:.4f}) | "
                f"Best ratio by RMSE: {self.ratio_best_rmse_row['Ratio']} ({self.ratio_best_rmse_row['RMSE']:.4f}) | "
                f"Best ratio by MSE: {self.ratio_best_mse_row['Ratio']} ({self.ratio_best_mse_row['MSE']:.4f}) | "
                f"Best ratio by MAE: {self.ratio_best_mae_row['Ratio']} ({self.ratio_best_mae_row['MAE']:.4f})"
            )

            if LIGHTGBM_FALLBACK_MESSAGE:
                self.status_text.set(f"{self.status_text.get()} | {LIGHTGBM_FALLBACK_MESSAGE}")

            if XGBOOST_FALLBACK_MESSAGE:
                self.status_text.set(f"{self.status_text.get()} | {XGBOOST_FALLBACK_MESSAGE}")

            messagebox.showinfo(
                "Run Complete",
                f"All selected models have finished running successfully!\n\nBest Model: {best_model['Model']} (R² = {best_model['R2']:.4f})"
            )

        except Exception as exc:
            messagebox.showerror("Run Failed", str(exc))
            self.status_text.set("Execution failed. Check dataset format and model selection.")

    def _update_results_table(self):
        for item in self.results_table.get_children():
            self.results_table.delete(item)

        for _, row in self.metrics_df.iterrows():
            self.results_table.insert(
                "",
                "end",
                values=(
                    row["Model"],
                    f"{row['R2']:.6f}",
                    f"{row['RMSE']:.6f}",
                    f"{row['MSE']:.6f}",
                    f"{row['MAE']:.6f}",
                ),
            )

    def _update_metric_charts(self):
        self._reset_canvas_hover_state(self.metrics_canvas)
        self._render_model_comparison_figure(self.metrics_fig)
        self._setup_canvas_zoom_selectors(self.metrics_canvas)
        self.metrics_canvas.draw_idle()

    def _update_ratio_chart(self):
        self._reset_canvas_hover_state(self.ratio_canvas)
        self._render_ratio_analysis_figure(self.ratio_fig)
        self._setup_canvas_zoom_selectors(self.ratio_canvas)
        self.ratio_canvas.draw_idle()

    def _update_prediction_chart(self):
        self._reset_canvas_hover_state(self.pred_canvas)
        self._render_prediction_figure(self.pred_fig)
        self._setup_canvas_zoom_selectors(self.pred_canvas)
        self.pred_canvas.draw_idle()

    def _update_combination_chart(self):
        self._reset_canvas_hover_state(self.combo_canvas)
        self._render_combination_figure(self.combo_fig)
        self._setup_canvas_zoom_selectors(self.combo_canvas)
        self.combo_canvas.draw_idle()

    def _update_output_preview(self, source_df: pd.DataFrame):
        self.output_text.delete("1.0", "end")

        _, _, target_col, feature_cols = self._get_features_and_target(source_df)
        best_row = self.metrics_df.iloc[0] if not self.metrics_df.empty else None

        header = [
            f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset: {self.dataset_path.get()}",
            f"Rows used: {len(source_df)}",
            f"Train-Test ratio used for detailed model run: {self._ratio_label(self.train_size_value)}",
            f"Features: {', '.join(feature_cols)}",
            f"Target: {target_col}",
            "",
            (
                f"Best metrics (highest R²): {best_row['Model']} | R²={best_row['R2']:.6f} | RMSE={best_row['RMSE']:.6f} | MSE={best_row['MSE']:.6f} | MAE={best_row['MAE']:.6f}"
                if best_row is not None
                else "Best metrics (highest R²): N/A"
            ),
            "",
            "Metrics (sorted by R² descending):",
            self.metrics_df.to_string(index=False),
            "",
            "Ratio analysis (best model across train-test ratios):",
            self.ratio_metrics_df[["Ratio", "R2", "RMSE", "MSE", "MAE"]].to_string(index=False) if not self.ratio_metrics_df.empty else "No ratio analysis available",
            "",
            (
                f"Best ratio by R²: {self.ratio_best_r2_row['Ratio']} (R²={self.ratio_best_r2_row['R2']:.6f})"
                if self.ratio_best_r2_row is not None
                else "Best ratio by R²: N/A"
            ),
            (
                f"Best ratio by RMSE: {self.ratio_best_rmse_row['Ratio']} (RMSE={self.ratio_best_rmse_row['RMSE']:.6f})"
                if self.ratio_best_rmse_row is not None
                else "Best ratio by RMSE: N/A"
            ),
            (
                f"Best ratio by MSE: {self.ratio_best_mse_row['Ratio']} (MSE={self.ratio_best_mse_row['MSE']:.6f})"
                if self.ratio_best_mse_row is not None
                else "Best ratio by MSE: N/A"
            ),
            (
                f"Best ratio by MAE: {self.ratio_best_mae_row['Ratio']} (MAE={self.ratio_best_mae_row['MAE']:.6f})"
                if self.ratio_best_mae_row is not None
                else "Best ratio by MAE: N/A"
            ),
            "",
            "Sample predictions (top 10 rows per selected model):",
            "",
        ]
        self.output_text.insert("end", "\n".join(header))

        for model_name, frame in self.prediction_frames.items():
            self.output_text.insert("end", f"[{model_name}]\n")
            self.output_text.insert("end", frame.head(10).to_string(index=False))
            self.output_text.insert("end", "\n\n")

        self.output_text.insert("end", "Top 10 best combinations per selected model:\n\n")
        for model_name, combo_df in self.top_combinations.items():
            self.output_text.insert("end", f"[{model_name}]\n")
            self.output_text.insert("end", combo_df[["Combination", "Predicted"]].to_string(index=False))
            self.output_text.insert("end", "\n\n")

    def _export_results(self):
        if self.metrics_df.empty:
            messagebox.showwarning("No Results", "Run at least one model before exporting.")
            return

        out_dir = Path("output") / "gui_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_path = out_dir / f"metrics_{stamp}.csv"
        self.metrics_df.to_csv(metrics_path, index=False)

        pred_path = out_dir / f"predictions_{stamp}.csv"
        merged = []
        for model_name, frame in self.prediction_frames.items():
            temp = frame.copy()
            temp.insert(0, "Model", model_name)
            merged.append(temp)
        pd.concat(merged, ignore_index=True).to_csv(pred_path, index=False)

        combo_path = out_dir / f"top10_combinations_{stamp}.csv"
        combo_merged = []
        for model_name, combo_df in self.top_combinations.items():
            temp = combo_df.copy()
            temp.insert(0, "Model", model_name)
            combo_merged.append(temp)
        if combo_merged:
            pd.concat(combo_merged, ignore_index=True).to_csv(combo_path, index=False)

        ratio_path = out_dir / f"ratio_metrics_{stamp}.csv"
        if not self.ratio_metrics_df.empty:
            self.ratio_metrics_df.to_csv(ratio_path, index=False)

        messagebox.showinfo("Export Complete", f"Saved:\n{metrics_path}\n{pred_path}\n{combo_path}\n{ratio_path}")
        self.status_text.set("Results exported in output/gui_runs.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelWorkbenchGUI(root)
    root.mainloop()
