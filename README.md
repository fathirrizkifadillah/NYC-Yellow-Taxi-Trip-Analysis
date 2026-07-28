# 🚕 NYC Yellow Taxi Trip Analysis — January 2026

End-to-end exploratory data analysis, geospatial visualization, and machine learning on **3.7 million** NYC Yellow Taxi trip records from January 2026.

> **Data Source:** [NYC Taxi & Limousine Commission (TLC) Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

## 📌 Project Highlights

- **3,724,889** raw trip records analyzed
- **14** EDA sub-analyses covering temporal, spatial, financial, and behavioral dimensions
- **3** geospatial choropleth maps (pickup demand, dropoff demand, median trip duration)
- **2** machine learning models compared for trip-duration prediction
- Comprehensive bilingual documentation (English & Bahasa Indonesia)

---

## 🔍 Key Findings

| Finding | Detail |
|---|---|
| **Peak demand** | ~6 PM daily; Saturday is the busiest day |
| **Manhattan dominance** | 3.1M+ trips start *and* end in Manhattan |
| **Short local trips** | 85% of trips stay within the same borough (median 1.56 mi) |
| **Interborough trips** | Only 15% of trips, but median distance 9.33 mi & median fare ~$55 |
| **Traffic pattern** | Fastest speeds at 3–5 AM; slowest during daytime congestion |
| **Payment** | Credit card dominates (~61%); Flex Fare is second-largest |
| **Tipping** | Lowest tip rates at 4–6 AM; highest in evening hours |
| **Best ML model** | CatBoost — MAE 5.50 min, RMSE 8.76 min, R² 0.633 |

---

## 📊 Sample Visualizations

<p align="center">
  <img src="img/EDA/NYC_Yellow_Taxi_Trip_Demand_Heatmap_by_Day_and_Hour.png" width="100%" alt="Trip Demand Heatmap by Day and Hour">
</p>

<p align="center">
  <img src="img/EDA/NYC_Yellow_Taxi_Tipping_Behavior_by_Pickup_Hour.png" width="100%" alt="Tipping Behavior by Pickup Hour">
</p>

<p align="center">
  <img src="img/geospatial/NYC_Yellow_Taxi_Pickup_Demand_by_Taxi_Zone.png" width="48%" alt="Pickup Demand Map">
  <img src="img/geospatial/Median_Yellow_Taxi_Trip_Duration_by_Pickup_Zone.png" width="48%" alt="Median Trip Duration Map">
</p>

<p align="center">
  <img src="img/ML/Model_Comparison.png" width="100%" alt="Model Comparison: Linear Regression vs CatBoost">
</p>

---

## 🗂️ Project Structure

```
Yellow_taxi_tripdata_NYCD/
│
├── main.ipynb                    # Main analysis notebook (123 cells)
├── Dataset_explanation.md        # Detailed dataset & column documentation (EN/ID)
├── taxi_zone_exp.md              # Taxi Zone lookup table documentation (EN/ID)
├── README.md
├── .gitignore
│
├── data/                         # (git-ignored)
│   ├── yellow_tripdata_2026-01.parquet    # Raw trip data (~61 MB)
│   ├── taxi_zone_lookup.csv               # Zone ID → Borough/Zone mapping
│   └── taxi_zones/                        # Shapefile for geospatial maps
│
├── img/
│   ├── EDA/                      # 13 EDA visualizations
│   ├── distribution/             # 2 distribution analysis plots
│   ├── geospatial/               # 3 choropleth maps
│   └── ML/                       # 4 ML evaluation & comparison plots
│
└── models/
    ├── linear_regression_trip_duration.pkl
    └── catboost_trip_duration.pkl
```

---

## 📒 Notebook Chapters

| Chapter | Content | Sections |
|---|---|---|
| **1 — Data Overview** | Load & inspect raw dataset structure | 1 |
| **2 — Data Quality Check** | Missing values, outliers, cleaning rules, feature engineering | 5 |
| **3 — Exploratory Data Analysis** | Temporal demand, spatial patterns, fare analysis, payment, tipping, interborough comparison | 14 |
| **4 — Geospatial Analysis** | Choropleth maps for pickup/dropoff demand & trip duration | 3 |
| **5 — Machine Learning** | Trip-duration prediction with Linear Regression & CatBoost | 8 |

Every analysis section includes a **markdown insight cell** explaining the findings.

---

## 🤖 Machine Learning

### Objective
Predict **trip duration (minutes)** using only pre-trip information.

### Approach
- **Train/test split:** Time-based (not random) to respect temporal ordering
- **Leakage prevention:** Post-trip fields (`fare_amount`, `total_amount`, `tip_amount`, `trip_distance`, etc.) excluded from features
- **Features:** `pickup_hour`, `pickup_dayofweek`, `pickup_zone`, `dropoff_zone`, `passenger_count`, `estimated_zone_distance_miles`

### Results

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| Linear Regression | 8.549 | 12.437 | 0.260 |
| **Enhanced CatBoost** ✅ | **5.501** | **8.760** | **0.633** |

CatBoost's top feature: **`estimated_zone_distance_miles`** (centroid-based zone-to-zone distance).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.11** | Core language |
| **Pandas** | Data manipulation & aggregation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical plot styling |
| **SciPy** | Skewness analysis |
| **GeoPandas** | Geospatial data handling & choropleth maps |
| **Scikit-learn** | Preprocessing pipeline, Linear Regression, metrics |
| **CatBoost** | Gradient boosting model for trip-duration prediction |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Yellow_taxi_tripdata_NYCD.git
cd Yellow_taxi_tripdata_NYCD
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy geopandas scikit-learn catboost
```

### 3. Download the dataset

Download the **Yellow Taxi Trip Records — January 2026** (Parquet) from the [NYC TLC website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and place it in `data/`:

```
data/yellow_tripdata_2026-01.parquet
```

The taxi zone lookup CSV and shapefile are already included in `data/`.

### 4. Run the notebook

```bash
jupyter notebook main.ipynb
```

---

## 📄 Documentation

- [**Dataset Explanation**](Dataset_explanation.md) — Full column descriptions, data quality summary, cleaning rules, derived features, suggested analysis questions, and ML scope (bilingual EN/ID)
- [**Taxi Zone Explanation**](taxi_zone_exp.md) — Zone lookup table documentation and join logic (bilingual EN/ID)

---

## 📜 License

This project uses publicly available data from the [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The analysis code is open for educational and portfolio purposes.
