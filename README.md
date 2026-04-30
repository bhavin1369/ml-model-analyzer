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

## Deploy on Render (Free)

### Step 1 — Push to GitHub

```bash
cd d:\AI\AI_project

git init
git add app.py templates/ requirements.txt render.yaml Procfile .gitignore README.md
git commit -m "ML Model Workbench - web app"
```

Create a new repository on [github.com/new](https://github.com/new), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/ml-model-workbench.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Render

1. Go to [render.com](https://render.com) and **sign up** (free with GitHub)
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect a repository"** → select your GitHub repo
4. Render auto-detects settings from `render.yaml`:
   - **Name:** `ml-model-workbench`
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
5. Select **Free** plan
6. Click **"Create Web Service"**
7. Wait 2–3 minutes for the build to finish
8. Your app is live at: `https://ml-model-workbench.onrender.com`

> **Note:** Free tier spins down after 15 minutes of inactivity. The first request after sleep takes ~30 seconds to wake up.

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
