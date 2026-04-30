import math
import importlib
import textwrap
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
DEFAULT_DATASET = Path("data.csv")

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
    try:
        xgb_module = importlib.import_module("xgboost")
        xgb_regressor_class = getattr(xgb_module, "XGBRegressor")
    except Exception as exc:
        raise ImportError(
            "XGBoost model requires the 'xgboost' package. "
            "Install it with: e:/ML/.venv/Scripts/python.exe -m pip install xgboost"
        ) from exc

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
    try:
        lgbm_module = importlib.import_module("lightgbm")
        lgbm_regressor_class = getattr(lgbm_module, "LGBMRegressor")
    except Exception as exc:
        raise ImportError(
            "LightGBM model requires the 'lightgbm' package. "
            "Install it with: e:/ML/.venv/Scripts/python.exe -m pip install lightgbm"
        ) from exc

    return lgbm_regressor_class(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=-1,
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
        self.root.configure(bg="#F5F7FA")

        self.dataset_path = tk.StringVar(value=str(DEFAULT_DATASET))
        self.test_size_value = 0.2
        self.status_text = tk.StringVar(value="Select dataset and model(s), then click Run Selected Models.")

        self.metrics_df = pd.DataFrame()
        self.prediction_frames = {}
        self.top_combinations = {}
        self.prediction_filter_frame = None
        self.combination_filter_frame = None
        self.prediction_view_vars = {}
        self.combination_view_vars = {}
        self.model_select_vars = {}

        self._configure_style()
        self._build_layout()
        self._populate_model_list()

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

        canvas_holder = ttk.Frame(container, padding=(8, 6, 8, 8))
        canvas_holder.pack(fill="both", expand=True)
        return container, canvas_holder, filter_frame

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

    def _configure_style(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Modern color palette
        primary_color = "#0D47A1"      # Deep professional blue
        primary_hover = "#1565C0"      # Lighter blue on hover
        accent_color ="#1565C0"       # Vibrant orange
        bg_color = "#F5F7FA"           # Light clean background
        fg_color = "#2C3E50"           # Dark text
        secondary_color = "#ECF0F3"    # Light gray-blue

        # Configure base theme
        style.configure("TFrame", background=bg_color)
        style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", font=("Segoe UI", 9), background=secondary_color)
        
        # Title styles
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=primary_color, background=bg_color)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 11, "bold"), foreground=primary_color, background=bg_color)
        style.configure("Info.TLabel", font=("Segoe UI", 9), foreground=fg_color, background=bg_color)
        
        # Control box styling
        style.configure("Control.TLabelframe", background=secondary_color, foreground=primary_color, borderwidth=1, relief="solid")
        style.configure("Control.TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground=primary_color, background=secondary_color)
        style.configure("ControlHeading.TLabel", font=("Segoe UI", 16, "bold"), foreground=primary_color, background=bg_color)
        style.configure("SectionTitle.TLabel", font=("Segoe UI", 10, "bold"), foreground=primary_color, background=bg_color)
        
        # Action button styling
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background=accent_color, foreground="white")
        style.map("Accent.TButton", background=[("active", "#1565C0"), ("pressed", "#F8F6F6")], foreground=[("active", "white")])
        
        # Primary button styling
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background=primary_color, foreground="white")
        style.map("Primary.TButton", background=[("active", primary_hover), ("pressed", "#0D3B7F")], foreground=[("active", "white")])
        
        # Treeview styling
        style.configure("Treeview", background="#FFFFFF", fieldbackground="#FFFFFF", foreground=fg_color, rowheight=25)
        style.configure("Treeview.Heading", background=primary_color, foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", secondary_color)], foreground=[("selected", primary_color)])
        
        # Entry styling
        style.configure("TEntry", fieldbackground="#FFFFFF", background=bg_color, foreground=fg_color, padding=5)
        style.map("TEntry", fieldbackground=[("focus", "#FFFFFF")], bordercolor=[("focus", primary_color)])

    def _build_layout(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="ML Model Workbench", style="Title.TLabel").grid(row=0, column=0, sticky="w", columnspan=6)

        ttk.Label(top, text="Dataset (CSV)", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 2))
        entry = ttk.Entry(top, textvariable=self.dataset_path, width=75)
        entry.grid(row=2, column=0, columnspan=4, sticky="we", padx=(0, 8))

        ttk.Button(top, text="Browse", command=self._browse_dataset).grid(row=2, column=4, sticky="we", padx=(0, 6))
        ttk.Button(top, text="Load Info", command=self._show_dataset_info).grid(row=2, column=5, sticky="we")

        ttk.Button(top, text="Run Selected Models", style="Accent.TButton", command=self.run_selected_models).grid(
            row=3, column=5, sticky="we", pady=(6, 0)
        )

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

        self.model_check_frame = tk.Frame(models_box, bg="#FFFFFF", bd=1, relief="solid")
        self.model_check_frame.pack(fill="both", expand=True)

        guide_box = ttk.LabelFrame(left_panel, text="QUICK START GUIDE", style="Control.TLabelframe", padding=8)
        guide_box.pack(fill="both", expand=True)
        guide_text_frame = ttk.Frame(guide_box)
        guide_text_frame.pack(fill="both", expand=True)

        self.guide_text = tk.Text(
            guide_text_frame,
            wrap="word",
            relief="solid",
            borderwidth=1,
            bg="#FFFFFF",
            fg="#2C3E50",
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

        charts_tab = ttk.Frame(notebook)
        predictions_tab = ttk.Frame(notebook)
        combinations_tab = ttk.Frame(notebook)
        table_tab = ttk.Frame(notebook)
        output_tab = ttk.Frame(notebook)

        notebook.add(charts_tab, text="Model Comparison")
        notebook.add(predictions_tab, text="Predicted vs Actual")
        notebook.add(combinations_tab, text="Top 10 Combinations")
        notebook.add(table_tab, text="Model Results")
        notebook.add(output_tab, text="Output Preview")

        _, metrics_holder, _ = self._build_graph_section(
            charts_tab,
            "Model Performance Comparison",
            self.open_metrics_fullscreen,
        )

        self.metrics_fig = Figure(figsize=(9, 6), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=metrics_holder)
        self.metrics_canvas.get_tk_widget().pack(fill="both", expand=True)

        _, predictions_holder, self.prediction_filter_frame = self._build_graph_section(
            predictions_tab,
            "Prediction Charts",
            self.open_predictions_fullscreen,
        )

        self.pred_fig = Figure(figsize=(9, 6), dpi=100)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=predictions_holder)
        self.pred_canvas.get_tk_widget().pack(fill="both", expand=True)

        _, combinations_holder, self.combination_filter_frame = self._build_graph_section(
            combinations_tab,
            "Top 10 Best Combinations by Model",
            self.open_combinations_fullscreen,
        )

        self.combo_fig = Figure(figsize=(9, 6), dpi=100)
        self.combo_canvas = FigureCanvasTkAgg(self.combo_fig, master=combinations_holder)
        self.combo_canvas.get_tk_widget().pack(fill="both", expand=True)

        table_frame = ttk.Frame(table_tab, padding=8)
        table_frame.pack(fill="both", expand=True)

        columns = ("Model", "RMSE", "MSE", "R2")
        self.results_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        for col in columns:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, anchor="center", width=160)
        self.results_table.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_table.yview)
        y_scroll.pack(side="right", fill="y")
        self.results_table.configure(yscrollcommand=y_scroll.set)

        self.output_text = tk.Text(output_tab, wrap="none", relief="solid", borderwidth=1, bg="#FFFFFF", fg="#2C3E50", font=("Courier New", 9))
        self.output_text.pack(fill="both", expand=True, padx=8, pady=8)

        status = ttk.Label(self.root, textvariable=self.status_text, style="Info.TLabel", padding=(12, 6))
        status.pack(fill="x", side="bottom")

        self._draw_empty_graphs()

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
        self._render_model_comparison_figure(self.metrics_fig, empty=True)
        self.metrics_canvas.draw_idle()

        self._render_prediction_figure(self.pred_fig, empty=True)
        self.pred_canvas.draw_idle()

        self._render_combination_figure(self.combo_fig, empty=True)
        self.combo_canvas.draw_idle()

    def _get_best_model_name(self):
        if self.metrics_df.empty:
            return None
        return str(self.metrics_df.iloc[0]["Model"])

    def _refresh_model_filters(self):
        model_names = self.metrics_df["Model"].tolist() if not self.metrics_df.empty else []

        if self.prediction_filter_frame is not None:
            self._create_view_checkboxes(
                self.prediction_filter_frame,
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

    def _on_combination_view_change(self, _event=None):
        self._update_combination_chart()

    def _build_metric_bar_colors(self, count, highlight_index, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        if count <= 1:
            return ["#FF6B35"]

        shades = np.linspace(0.42, 0.88, count)
        colors = [cmap(value) for value in shades]
        colors[highlight_index] = "#FF6B35"
        return colors

    def _annotate_horizontal_bars(self, ax, values, offset=0.01, fmt="{:.4f}", fontsize=8):
        if len(values) == 0:
            return

        max_value = float(np.max(values)) if np.size(values) else 0.0
        label_offset = max(max_value * offset, 0.001)
        for bar, value in zip(ax.patches, values):
            ax.text(
                bar.get_width() + label_offset,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=fontsize,
            )

    def _annotate_vertical_bars(self, ax, values, offset=0.002, fmt="{:.4f}", fontsize=8):
        if len(values) == 0:
            return

        max_value = float(np.max(values)) if np.size(values) else 0.0
        label_offset = max(max_value * offset, 0.001)
        for bar, value in zip(ax.patches, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    def _render_model_comparison_figure(self, figure, empty=False):
        figure.clear()

        if empty or self.metrics_df.empty:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "Run models to see the model comparison dashboard", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return

        display_df = self.metrics_df.sort_values("R2", ascending=True).reset_index(drop=True)
        models = display_df["Model"].tolist()
        rmse_vals = display_df["RMSE"].to_numpy(dtype=float)
        r2_vals = display_df["R2"].to_numpy(dtype=float)

        best_idx = len(display_df) - 1
        best_row = display_df.iloc[best_idx]

        ax1 = figure.add_subplot(121)
        ax1.barh(
            models,
            r2_vals * 100.0,
            color=self._build_metric_bar_colors(len(models), best_idx, "Blues"),
            edgecolor="#0D47A1",
            linewidth=0.3,
        )
        self._annotate_horizontal_bars(ax1, r2_vals * 100.0, offset=0.006, fmt="{:.2f}%", fontsize=8)
        ax1.axvline(best_row["R2"] * 100.0, color="#FF6B35", linestyle="--", linewidth=1.6, label="This Model")
        ax1.set_title(
            f"Model Accuracy Comparison\n{best_row['Model']}: {best_row['R2'] * 100:.2f}%",
            fontweight="bold",
            fontsize=12,
        )
        ax1.set_xlabel("R² Score (%)", fontweight="bold")
        ax1.set_xlim(0, max(100.0, float(np.max(r2_vals) * 100.0) * 1.05))
        ax1.tick_params(axis="y", labelsize=9)
        ax1.legend(loc="upper right", fontsize=8)

        ax2 = figure.add_subplot(122)
        ax2.barh(
            models,
            rmse_vals,
            color=self._build_metric_bar_colors(len(models), best_idx, "Oranges"),
            edgecolor="#0D47A1",
            linewidth=0.3,
        )
        self._annotate_horizontal_bars(ax2, rmse_vals, offset=0.02, fmt="{:.4f}", fontsize=8)
        ax2.axvline(best_row["RMSE"], color="#FF6B35", linestyle="--", linewidth=1.6, label="This Model")
        ax2.set_title(
            f"Model Error Comparison\n{best_row['Model']}: {best_row['RMSE']:.6f}",
            fontweight="bold",
            fontsize=12,
        )
        ax2.set_xlabel("RMSE (Error Rate)", fontweight="bold")
        ax2.set_xlim(0, max(float(np.max(rmse_vals)) * 1.05, float(best_row["RMSE"]) * 1.1, 0.01))
        ax2.tick_params(axis="y", labelsize=9)
        ax2.legend(loc="lower right", fontsize=8)

        for ax in (ax1, ax2):
            ax.grid(axis="x", linestyle="--", alpha=0.2)
            ax.invert_yaxis()

        figure.tight_layout()

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

        all_actual = np.concatenate([self.prediction_frames[name]["Actual"].values for name in names])
        all_pred = np.concatenate([self.prediction_frames[name]["Predicted"].values for name in names])
        low = float(min(np.min(all_actual), np.min(all_pred)))
        high = float(max(np.max(all_actual), np.max(all_pred)))
        pad = 0.05 * (high - low if high != low else 1.0)
        low, high = low - pad, high + pad

        for idx, name in enumerate(names, start=1):
            ax = figure.add_subplot(rows, cols, idx)
            frame = self.prediction_frames[name]
            ax.scatter(frame["Actual"], frame["Predicted"], s=18, alpha=0.8, color="#457b9d")
            ax.plot([low, high], [low, high], "r--", linewidth=1.2)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            metric_row = self.metrics_df[self.metrics_df["Model"] == name].iloc[0]
            ax.set_title(f"{name} | R²={metric_row['R2']:.4f} | RMSE={metric_row['RMSE']:.4f}")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

        figure.tight_layout()

    def _format_feature_value(self, value):
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            if abs(value) >= 10000 or (0 < abs(value) < 0.001):
                return f"{value:.2e}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
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

            grid = figure.add_gridspec(1, 2, width_ratios=[1.40, 1.30], wspace=0.04)
            ax_bar = figure.add_subplot(grid[0, 0])
            ax_text = figure.add_subplot(grid[0, 1])

            labels = [f"C{i + 1}" for i in range(len(top_df))]
            values = top_df["Predicted"].to_numpy(dtype=float)
            colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(values)))

            bars = ax_bar.bar(labels, values, color=colors, edgecolor="none")
            self._annotate_vertical_bars(ax_bar, values, offset=0.004, fmt="{:.4f}", fontsize=9)
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

            for bar, val in zip(bars, values):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    val,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

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
            ax.barh(ranks[::-1], values, color="#FF6B35")
            self._annotate_horizontal_bars(ax, values, offset=0.01, fmt="{:.4f}", fontsize=8)
            best_value = float(top_df["Predicted"].iloc[0])
            ax.set_title(f"{model_name} - Top 10 (Best={best_value:.4f})")
            ax.set_xlabel("Predicted Target")
            ax.tick_params(axis="y", labelsize=9)

        figure.tight_layout()

    def _open_fullscreen_window(self, title, render_function):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.attributes("-fullscreen", True)
        window.configure(background="#F5F7FA")

        def close_window(_event=None):
            window.destroy()

        window.bind("<Escape>", close_window)

        wrapper = ttk.Frame(window, padding=10)
        wrapper.pack(fill="both", expand=True)

        topbar = ttk.Frame(wrapper)
        topbar.pack(fill="x", pady=(0, 8))

        ttk.Label(topbar, text=title, style="Title.TLabel").pack(side="left")
        ttk.Button(topbar, text="Close", command=close_window).pack(side="right")

        fig = Figure(figsize=(14, 9), dpi=110)
        render_function(fig)
        canvas = FigureCanvasTkAgg(fig, master=wrapper)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()

        def on_resize(event):
            if event.width < 200 or event.height < 150:
                return
            fig.set_size_inches(event.width / fig.dpi, event.height / fig.dpi, forward=True)
            render_function(fig)
            canvas.draw_idle()

        wrapper.bind("<Configure>", on_resize)

    def open_metrics_fullscreen(self):
        self._open_fullscreen_window("Model Comparison Full Screen", self._render_model_comparison_figure)

    def open_predictions_fullscreen(self):
        self._open_fullscreen_window("Prediction Charts Full Screen", self._render_prediction_figure)

    def open_combinations_fullscreen(self):
        self._open_fullscreen_window("Top 10 Combinations Full Screen", self._render_combination_figure)

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

        feature_cols = dataset.columns[:-1].tolist()
        target_col = dataset.columns[-1]

        msg = (
            f"Rows: {len(dataset)}\n"
            f"Columns: {len(dataset.columns)}\n"
            f"Target column: {target_col}\n"
            f"Feature columns: {', '.join(feature_cols)}"
        )
        messagebox.showinfo("Dataset Information", msg)
        self.status_text.set("Dataset loaded successfully.")

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

    def run_selected_models(self):
        selected_models = self._get_selected_model_names()
        if not selected_models:
            messagebox.showwarning("No Model Selected", "Please select at least one model.")
            return

        try:
            df = self._load_dataset(self.dataset_path.get())
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=float(self.test_size_value),
                random_state=RANDOM_STATE,
            )

            records = []
            predictions = {}
            top_combinations = {}
            feature_cols = X.columns.tolist()

            for model_name in selected_models:
                model = MODEL_BUILDERS[model_name]()
                # Keep preprocessing with each model to avoid data leakage and keep behavior consistent.
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
                    "RMSE": rmse,
                    "MSE": mse,
                    "R2": r2,
                })

                pred_df = pd.DataFrame({
                    "Actual": y_test.reset_index(drop=True),
                    "Predicted": pd.Series(y_pred),
                    "Residual": y_test.reset_index(drop=True) - pd.Series(y_pred),
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

            self.metrics_df = pd.DataFrame(records).sort_values("R2", ascending=False).reset_index(drop=True)
            self.prediction_frames = predictions
            self.top_combinations = top_combinations
            self._refresh_model_filters()

            self._update_results_table()
            self._update_metric_charts()
            self._update_prediction_chart()
            self._update_combination_chart()
            self._update_output_preview(df)

            best_model = self.metrics_df.iloc[0]
            self.status_text.set(
                f"Run complete. Best model: {best_model['Model']} (R²={best_model['R2']:.4f}, RMSE={best_model['RMSE']:.4f})"
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
                    f"{row['RMSE']:.6f}",
                    f"{row['MSE']:.6f}",
                    f"{row['R2']:.6f}",
                ),
            )

    def _update_metric_charts(self):
        self._render_model_comparison_figure(self.metrics_fig)
        self.metrics_canvas.draw_idle()

    def _update_prediction_chart(self):
        self._render_prediction_figure(self.pred_fig)
        self.pred_canvas.draw_idle()

    def _update_combination_chart(self):
        self._render_combination_figure(self.combo_fig)
        self.combo_canvas.draw_idle()

    def _update_output_preview(self, source_df: pd.DataFrame):
        self.output_text.delete("1.0", "end")

        header = [
            f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset: {self.dataset_path.get()}",
            f"Rows used: {len(source_df)}",
            f"Features: {', '.join(source_df.columns[:-1])}",
            f"Target: {source_df.columns[-1]}",
            "",
            "Metrics (sorted by R² descending):",
            self.metrics_df.to_string(index=False),
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

        messagebox.showinfo("Export Complete", f"Saved:\n{metrics_path}\n{pred_path}\n{combo_path}")
        self.status_text.set("Results exported in output/gui_runs.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelWorkbenchGUI(root)
    root.mainloop()
