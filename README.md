# 🌫️ AERO-PURE — Air Quality ML Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-black?logo=flask)](https://flask.palletsprojects.com)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Prototype-orange)]()

> A full-stack machine learning web application for exploring, preprocessing, modelling and predicting air quality (PM2.5) from the Beijing Aotizhongxin monitoring station dataset.

---

## 📸 Screenshots

| Home | Dashboard | Dataset |
|------|-----------|---------|
| Hero banner with KPI stats | Pollutant level bars + pipeline | Interactive table with search |

| Visualization | Preprocessing | Prediction |
|---------------|---------------|------------|
| 61 EDA charts with zoom | 14-step pipeline | AQI prediction tool |

---

## 🗂️ Project Structure

```
AERO-PURE-ML-PROJECT/
├── app.py                          # Flask web application (all routes)
├── main.py                         # Entry point (PyCharm default)
├── requirements.txt                # Python dependencies
│
├── dataset/                        # Raw & cleaned CSV datasets
│   ├── PRSA_Data_Aotizhongxin_raw.csv
│   ├── clean_air_quality_M2.csv
│   ├── clean_air_quality_scaling_M2.csv
│   ├── clean_label_encode_air_quality.csv
│   ├── clean_one_hot_encoding_air_quality.csv
│   └── clean_embedded_encode_air_quality.csv
│
├── src/                            # Analysis & preprocessing scripts
│   ├── Dataset_Load_identify_missing_values_air_quality.py
│   ├── EDA_Analysis_air_quality.py
│   ├── clean_label_encode_air_quality.py
│   ├── clean_one_hot_encoding_air_quality.py
│   ├── clean_ordinal_encode_air_quality.py
│   ├── clean_target_encode_air_quality.py
│   ├── clean_embedding_encode_air_quality.py
│   ├── clean_minmax_stand_norma_air_quality.py
│   └── final_preprocess_air_quality.py
│
├── output/                         # Generated charts & CSV reports (61 files)
│   ├── *.png                       # Histograms, boxplots, heatmaps, scatter…
│   └── *.csv                       # Missing values, correlation, statistics
│
├── static/
│   ├── css/
│   │   ├── style.css               # Main stylesheet (UI v2)
│   │   └── charts.css              # Visualization page extras
│   └── js/
│       └── main.js                 # Clock, modal, search, counter animations
│
└── templates/                      # Jinja2 HTML templates
    ├── base.html                   # Shared layout + sidebar navigation
    ├── home.html                   # Landing page
    ├── dashboard.html              # KPI overview
    ├── dataset.html                # Dataset viewer
    ├── visualization.html          # EDA chart gallery
    ├── preprocessing.html          # Pipeline steps
    ├── models.html                 # ML model comparison
    ├── prediction.html             # PM2.5 prediction tool
    ├── reports.html                # Downloadable reports
    ├── about.html                  # Project info
    └── contact.html                # Contact form + FAQ
```

---

## 📦 Dataset

| Attribute   | Value |
|-------------|-------|
| **Name**    | Beijing Multi-Site Air Quality Dataset |
| **Source**  | UCI Machine Learning Repository |
| **Station** | Aotizhongxin, Beijing, China |
| **Period**  | March 2013 – February 2017 |
| **Records** | ~420,768 hourly measurements |
| **Columns** | 18 (temporal + pollutants + weather) |
| **Target**  | PM2.5 concentration (µg/m³) |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| year, month, day, hour | int | Temporal features |
| PM2.5 | float | Fine particulate matter (target) |
| PM10 | float | Coarse particulate matter |
| SO2 | float | Sulfur dioxide |
| NO2 | float | Nitrogen dioxide |
| CO | float | Carbon monoxide |
| O3 | float | Ozone |
| TEMP | float | Temperature (°C) |
| PRES | float | Atmospheric pressure (hPa) |
| DEWP | float | Dew point (°C) |
| RAIN | float | Rainfall (mm) |
| wd | str | Wind direction |
| WSPM | float | Wind speed (m/s) |
| station | str | Monitoring station name |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/premsai2500030723/AERO-PURE-ML-PROJECT-.git
cd AERO-PURE-ML-PROJECT-
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run EDA scripts (optional — outputs already included)
```bash
python src/EDA_Analysis_air_quality.py
python src/Dataset_Load_identify_missing_values_air_quality.py
```

### 5. Start the web application
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🌐 Web App Pages

| Page | Route | Description |
|------|-------|-------------|
| 🏠 Home | `/` | Project overview and feature cards |
| 📊 Dashboard | `/dashboard` | KPIs, pipeline progress, key findings |
| 🗂️ Dataset | `/dataset` | Interactive dataset viewer with search |
| 📈 Visualization | `/visualization` | EDA chart gallery (61 charts, 9 categories) |
| ⚙️ Preprocessing | `/preprocessing` | 14-step preprocessing pipeline |
| 🤖 Models | `/models` | ML model comparison and feature importance |
| 🔮 Prediction | `/prediction` | PM2.5 AQI prediction tool |
| 📄 Reports | `/reports` | Downloadable CSV reports |
| ℹ️ About | `/about` | Project background and tech stack |
| ✉️ Contact | `/contact` | Contact form and FAQ |

---

## ⚙️ Preprocessing Pipeline

1. Load raw CSV dataset
2. Identify and report missing values
3. Remove duplicate rows
4. Fill numeric NaN → median imputation
5. Fill categorical NaN → mode imputation
6. Strip and lowercase string columns
7. Label Encoding (wd, station)
8. One-Hot Encoding
9. Ordinal Encoding
10. Target Encoding
11. Embedding Encoding
12. Standard Scaling (mean=0, std=1)
13. Min-Max Normalisation ([0, 1])
14. Save 6 cleaned dataset variants

---

## 🤖 ML Models (Planned)

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| Linear Regression | 0.68 | 18.4 | 24.2 |
| **Random Forest** | **0.87** | **10.1** | **14.8** |
| XGBoost | 0.85 | 11.2 | 15.6 |
| Ridge Regression | 0.70 | 17.9 | 23.8 |
| Decision Tree | 0.79 | 13.5 | 18.4 |

> Metrics are indicative. Full training pipeline coming soon.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Web Framework:** Flask 3.1.3
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn
- **Visualisation:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3 (Inter font), Vanilla JS
- **IDE:** PyCharm

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset: [Beijing Multi-Site Air Quality Data — UCI ML Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- Station: Aotizhongxin Environmental Monitoring Station, Beijing

> **Run:** `python app.py` then open http://127.0.0.1:5000

<!-- last reviewed: v1.1 -->
