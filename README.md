# ML Model Workbench

A web-based machine learning workbench for training, comparing, and visualizing regression models. Upload any CSV dataset, select from 11 ML algorithms, and get interactive research-paper quality charts — all from your browser.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **11 Regression Models** — Linear Regression, Decision Tree, Random Forest, Bagging, Gradient Boosting, KNN, SVR, Extra Trees, AdaBoost, LightGBM, XGBoost
- **Interactive Charts** — Plotly-powered visualizations with zoom, pan, and hover
- **Model Comparison** — Side-by-side R² and RMSE bar charts with viridis color scale
- **Predicted vs Actual** — Scatter plots with residual color mapping (Red-Yellow-Green)
- **Top 10 Combinations** — Best input combinations per model with rainbow viridis bars
- **Model Filtering** — Toggle individual models on/off for prediction and combination views
- **Drag & Drop Upload** — CSV file upload with instant dataset preview
- **Research Paper Ready** — Clean white charts with academic styling

---

## Project Structure

```
AI_project/
├── app.py                 # Flask web server (main application)
├── templates/
│   └── index.html         # Web UI (HTML + CSS + JavaScript)
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
├── Procfile               # Process definition for hosting
├── .gitignore             # Git ignore rules
├── model_gui.py           # Original Tkinter desktop version
└── uploads/               # Uploaded CSV files (auto-created)
```

---

## Run Locally

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-model-workbench.git
cd ml-model-workbench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open http://localhost:5000 in your browser.

---

## How to Use

1. **Upload** — Click the upload area or drag a CSV file (must have headers, last column = target)
2. **Select Models** — Check/uncheck models in the sidebar
3. **Train** — Click "Train Models" and wait for results
4. **View Results** — Use tabs to switch between:
   - **Comparison** — R² and RMSE bar charts for all models
   - **Predicted vs Actual** — Scatter plots per model (filter with chips)
   - **Top 10 Combinations** — Best input combinations per model (filter with chips)
   - **Results Table** — Sorted metrics table

---

## Dataset Format

Your CSV file should look like this:

| feature_1 | feature_2 | feature_3 | target |
|-----------|-----------|-----------|--------|
| 0.5       | 1.2       | 3.0       | 0.85   |
| 0.8       | 2.1       | 1.5       | 0.92   |
| ...       | ...       | ...       | ...    |

- First row must be **column headers**
- Last column is the **target variable** (what you want to predict)
- All values should be **numeric**
- At least **10 rows** required

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend   | Python, Flask |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Charts    | Plotly.js |
| Frontend  | HTML, CSS, JavaScript |
| Hosting   | Render (free tier) |
| Server    | Gunicorn |

---

## License

MIT License — free to use for academic and personal projects.
