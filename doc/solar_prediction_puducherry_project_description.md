# Pudhuvai SolarScope

## 1. Project Overview

**Pudhuvai SolarScope** is an intermediate-level machine learning and geospatial application that estimates daily solar radiation for the four geographically separated regions of the Union Territory of Puducherry: Puducherry, Karaikal, Mahe and Yanam.

The system combines historical solar-radiation data from **NASA POWER** with historical and live weather forecasts from **Open-Meteo**. A supervised machine learning model is trained using archived weather forecasts as input features and NASA POWER solar-radiation values as the target. Once trained, the model uses current weather forecasts to estimate solar radiation for the next three days.

The predictions are presented through an interactive administrative map and a regional dashboard. Users can select a region and, where suitable administrative boundary data is available, a subdivision or taluk to view its forecast, historical patterns and an indicative rooftop solar-energy estimate.

The project is intended as a decision-support and educational tool. It does not provide a guarantee of actual rooftop electricity generation, nor does it replace measurements from local solar monitoring stations.

## 2. Problem Statement

Solar-energy availability varies with cloud cover, rainfall, humidity, temperature, season and other atmospheric conditions. People exploring rooftop solar installations or studying renewable-energy potential benefit from accessible, location-oriented information about expected solar radiation.

However, general weather forecasts do not always translate directly into an easy-to-interpret estimate of solar-energy availability. In addition, the four regions of Puducherry are geographically separated and have different local weather patterns. Puducherry and Karaikal lie on the Coromandel coast and receive most of their rain from the north-east monsoon (October–December); Mahe lies on the Malabar coast and is heavily affected by the south-west monsoon (June–September); Yanam, in the Godavari delta at a higher latitude, is affected by the south-west monsoon and the post-monsoon cyclone season. Their cloudy, low-radiation periods therefore occur at different times of year, and a model that combines all four regions without accounting for location could reduce prediction accuracy.

This project addresses the problem by developing a location-aware system that:
- estimates daily solar radiation for the next three days;
- uses forecast information available before each prediction date;
- presents predictions in a simple map-based interface;
- allows users to compare regions and inspect historical trends; and
- provides an indicative calculation of possible rooftop solar energy based on user-entered system capacity.

## 3. Aim

To develop a machine learning-based, three-day solar-radiation prediction and geospatial decision-support system for the four regions of the Union Territory of Puducherry using NASA POWER historical solar data and Open-Meteo weather forecasts.

## 4. Objectives

1. Collect historical daily solar-radiation data for representative locations in Puducherry, Karaikal, Mahe and Yanam using NASA POWER.
2. Obtain archived weather forecasts from Open-Meteo to construct realistic historical training examples.
3. Prepare a combined dataset by matching forecast inputs with the corresponding NASA POWER solar-radiation target by date and location.
4. Preprocess the data, handle missing values, create useful date and lead-time features, and examine seasonal patterns.
5. Train and compare regression models, using a simple baseline and models such as Random Forest and XGBoost.
6. Evaluate the model using chronological training, validation and test periods, with MAE, RMSE and R².
7. Integrate the trained model with the live Open-Meteo Forecast API to estimate solar radiation for the next three days.
8. Build an interactive map-based interface for exploring the four regions and their available administrative subdivisions.
9. Display predictions, historical trends, regional comparisons and an indicative rooftop-energy calculator.
10. Document the data limitations, model performance, assumptions and possible future improvements.

## 5. Geographic Scope

The application covers all four regions of the Union Territory:

- **Puducherry**
- **Karaikal**
- **Mahe**
- **Yanam**

The map should first display the Union Territory's four geographically separated regions. Selecting a region opens its regional view. If reliable taluk or subdivision polygon boundaries are available, users can select those areas as well.

Administrative boundary files must represent the relevant administrative units. Electoral constituency boundaries should not be treated as taluk boundaries unless their equivalence has been independently verified.

### Spatial-resolution limitation

NASA POWER provides gridded estimates rather than a separate ground measurement for every taluk or neighbourhood. Its solar-radiation product has a relatively coarse spatial resolution (approximately 1°). Consequently, nearby subdivisions may fall within the same grid cell and receive identical or very similar values. This is a data-resolution limitation, not a reason to manufacture differences between areas.

The application should:
- record the coordinates used for each representative location;
- identify when multiple administrative areas use the same underlying grid cell, where this can be established;
- avoid presenting repeated values as independent local measurements; and
- describe higher-resolution satellite products or local weather-station measurements as possible future enhancements.

## 6. Data Sources and Their Roles

### 6.1 NASA POWER – historical solar-radiation target

NASA POWER's Daily Point API provides historical daily solar and meteorological data for specified coordinates.

**Primary target variable**

| Variable | Description | Role |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave downward radiation | Target variable: daily solar radiation |

The target is expressed in kWh/m²/day in the NASA POWER daily product.

Additional historical meteorological variables may be collected for exploratory analysis and climatological comparisons, for example:
- `T2M` – temperature at 2 metres;
- `T2M_MAX` – daily maximum temperature at 2 metres;
- `T2M_MIN` – daily minimum temperature at 2 metres;
- `RH2M` – relative humidity at 2 metres;
- `PRECTOTCORR` – corrected precipitation; and
- `WS10M` – wind speed at 10 metres.

These historical observed weather values must not be substituted directly for forecast inputs in the forecast-assisted model. They can support exploratory analysis, descriptive charts and historical baselines.

### 6.2 Open-Meteo Previous Runs API – archived forecast inputs

The Previous Runs API provides archived weather forecasts associated with earlier model runs and fixed forecast lead times. It is used to obtain weather information that would have been available before the target date.

Potential input variables include:
- forecast shortwave radiation;
- forecast temperature;
- forecast relative humidity;
- forecast cloud cover;
- forecast precipitation; and
- forecast wind speed.

The exact variables and forecast model must be checked for availability across all four locations and all three lead times before the final dataset is built.

**Audit result (Nov 2026, `results/archive_audit/AUDIT_REPORT.md`):**
- The pinned model is **`ecmwf_ifs025`**. It has complete coverage of all six variables and lead days 1–3 from March 2024 for all four regions, with a single 6-day outage on 17–22 April 2026.
- `best_match` was rejected because Open-Meteo chooses its underlying model automatically and may change it.
- `icon_seamless` was rejected because ICON Global is discontinued on 30 Nov 2026.
- The other models either had shorter archives or larger errors.

**Lead-time definition:** `<variable>_previous_dayN` is defined per hour: each value was predicted N × 24 hours before that hour. A daily "lead N" aggregate is therefore assembled from several model runs. Its nominal issue date is target date − N days.

### 6.3 Open-Meteo Forecast API – live prediction inputs

The Forecast API supplies current forecasts for future dates. At application runtime, the system requests the same weather variables and applies the same daily aggregation and feature transformations used for the archived training forecasts.

The project should use a consistent forecast model/source wherever possible. The model identifier, forecast issue/run time, retrieval time and location coordinates should be stored with prediction records.

### 6.4 Source and licensing notes

NASA POWER data and Open-Meteo services have their own documentation, attribution and usage conditions. The project report and application should credit both providers. Open-Meteo's free API is intended for non-commercial use under its published terms; those terms should be checked again before public or commercial deployment.

## 7. Machine Learning Approach

### 7.1 Forecast-assisted supervised learning

The model is designed as a **solar-radiation forecast correction/estimation model**. It learns the relationship between weather forecasts issued in advance and the later NASA POWER solar-radiation value for the corresponding date and location.

It is important not to train the model only on historical observed weather and then feed it live weather forecasts. Observations and forecasts can have different statistical characteristics. Training and inference inputs should be constructed in a consistent way.

### 7.2 Training-row construction

For each location and target date, create historical examples for forecast lead times of one, two and three days.

For example, for a target date of 15 November:

| Forecast issue date | Target date | Lead time | Input data | Target |
|---|---|---:|---|---|
| 14 November | 15 November | 1 day | Archived forecast issued on 14 November | NASA solar value for 15 November |
| 13 November | 15 November | 2 days | Archived forecast issued on 13 November | NASA solar value for 15 November |
| 12 November | 15 November | 3 days | Archived forecast issued on 12 November | NASA solar value for 15 November |

The same eventual NASA target may therefore appear in three rows, one for each lead time. This is intentional. During evaluation, all rows belonging to the same target date must remain in the same data partition.

### 7.3 Model input features

A proposed feature set is:

| Feature | Meaning |
|---|---|
| `forecast_shortwave_radiation` | Open-Meteo forecast solar/shortwave radiation, converted to the chosen target unit |
| `forecast_temperature_mean` | Mean forecast temperature for the target date |
| `forecast_temperature_max` | Forecast maximum temperature |
| `forecast_temperature_min` | Forecast minimum temperature |
| `forecast_humidity_mean` | Mean forecast relative humidity |
| `forecast_cloud_cover_mean` | Mean forecast cloud cover |
| `forecast_precipitation_sum` | Forecast daily precipitation |
| `forecast_wind_speed_mean` | Mean forecast wind speed |
| `lead_days` | Forecast lead time: 1, 2 or 3 |
| `doy_sin`, `doy_cos` | Cyclical encoding of the target date's day of year (sin/cos of 2π·doy/365.25), so that late December and early January are treated as adjacent |
| `climatology_solar_radiation` | Region-specific long-term mean NASA `ALLSKY_SFC_SW_DWN` for the target day of year (smoothed), computed only from years **before** the training period |
| `clear_sky_radiation` (optional) | Region-specific clear-sky radiation climatology (e.g. NASA `CLRSKY_SFC_SW_DWN`) for the target day of year; normalises the latitude difference between regions |
| `region_id` | Region identifier: one-hot encoded for linear models, categorical for tree-based models |

The final features should be limited to variables that are available consistently in both archived forecasts and the live Forecast API. Do not include a feature in training if it cannot be supplied at inference time. The climatology features are precomputed lookup tables per region and day of year, so they are available at inference time.

Latitude and longitude are recorded as metadata but are not used as model features: with only four fixed representative points they carry the same information as `region_id` and cannot generalise to new locations.

**Exclusion found in Module 6:** `forecast_cloud_cover_mean` is excluded from the final model. From late 2025, ECMWF's archived cloud-cover forecasts rose by 15–35 percentage points in every month compared with a year earlier, while forecast and NASA radiation did not fall. This is an inconsistency in how the input is produced, not a real change in the weather, and it made a model trained on 2024–25 under-predict (see `notebooks/04_model_experiments.ipynb` §4).

### 7.4 Target variable

`actual_solar_radiation` is the NASA POWER `ALLSKY_SFC_SW_DWN` value for the target date and matching representative location.

In this project, “actual” means the retrospective NASA POWER value used as the reference label. It is a gridded estimate, not a direct local pyranometer observation.

### 7.5 Candidate models

- **Baseline model:** use Open-Meteo's own forecast shortwave radiation after unit conversion. A monthly historical average can also be included as a simple reference.
- **Linear Regression:** simple interpretable regression baseline.
- **Random Forest Regressor:** captures nonlinear relationships and interactions with limited feature scaling requirements.
- **XGBoost Regressor:** candidate primary model for tabular forecast features.

Deep learning or LSTM models are optional and are not required for the core project. The final model should be selected based on chronological test performance rather than model complexity.

### 7.6 Regional modelling strategy

Because the four regions have different seasonal weather regimes (see Section 2), location must be accounted for explicitly.

**Primary approach: one pooled, region-aware model.** A single model is trained on rows from all four regions, with `region_id`, region-specific climatology and cyclical seasonal features as inputs.

Rationale:
- The archived forecast record covers only a few years, so each region contributes only a few monsoon cycles. Separate per-region models would each be trained on very little data and would be prone to overfitting.
- Pooling allows the model to learn relationships common to all regions (for example, higher forecast cloud cover → lower radiation), while the region and climatology features capture region-specific behaviour and region-specific forecast bias.

Implementation notes:
- Tree-based models (Random Forest, XGBoost) learn region × feature interactions automatically.
- For Linear Regression, include explicit interaction terms between `region_id` and key forecast features (e.g. forecast shortwave radiation, cloud cover); otherwise region only shifts the intercept.
- **Optional residual target:** instead of predicting NASA radiation directly, the model may predict the correction `actual_solar_radiation − forecast_shortwave_radiation`, or a clearness-index ratio relative to clear-sky radiation. This frames the task as region-specific forecast correction and avoids tree models' inability to extrapolate. The final prediction is reconstructed by adding the correction back to the raw forecast.

The choice of the pooled region-aware model must be validated against alternatives (Section 10.3) rather than assumed.

## 8. Dataset Design

The project should maintain separate raw datasets and a final model-ready dataset.

### Dataset A: `nasa_solar_history.csv`

One row per representative location and date.

| Column | Description |
|---|---|
| `date` | Observation date |
| `region_id` | Puducherry, Karaikal, Mahe or Yanam |
| `latitude` | Requested latitude |
| `longitude` | Requested longitude |
| `ALLSKY_SFC_SW_DWN` | NASA POWER daily solar-radiation target |
| `T2M`, `T2M_MAX`, `T2M_MIN` | Optional historical temperature variables |
| `RH2M` | Optional historical relative humidity |
| `PRECTOTCORR` | Optional historical precipitation |
| `WS10M` | Optional historical wind speed |

### Dataset B: `archived_forecasts.csv`

One row per location, target date and forecast lead time after daily aggregation.

| Column | Description |
|---|---|
| `forecast_issue_date` | Nominal issue date (target date − lead days); see the lead-time definition in Section 6.2 |
| `target_date` | Date for which the forecast applies (IST calendar day) |
| `lead_days` | 1, 2 or 3 |
| `region_id` | Region identifier |
| `latitude`, `longitude` | Requested coordinates |
| `grid_latitude`, `grid_longitude` | Grid cell returned by Open-Meteo |
| `hours_complete` | Minimum number of hourly values present across variables (24 = complete day) |
| `forecast_model` | Weather model/source used |
| `forecast_shortwave_radiation` | Forecast daily solar radiation |
| `forecast_temperature_mean` | Mean forecast temperature |
| `forecast_temperature_max` | Forecast maximum temperature |
| `forecast_temperature_min` | Forecast minimum temperature |
| `forecast_humidity_mean` | Mean forecast humidity |
| `forecast_cloud_cover_mean` | Mean forecast cloud cover |
| `forecast_precipitation_sum` | Daily forecast precipitation |
| `forecast_wind_speed_mean` | Mean forecast wind speed |

### Dataset C: `solar_model_training.csv`

Join Dataset B to Dataset A using `target_date` and `region_id`, after verifying that the coordinates refer to the intended matching locations.

It contains:
- the forecast features;
- the lead time;
- the region, cyclical season and region-specific climatology features; and
- `actual_solar_radiation`, the NASA POWER target.

Only rows with valid forecast features and a valid target should be included in the supervised training dataset. Keep the raw source files unchanged so the joining and cleaning steps can be reproduced.

### Historical coverage

NASA POWER's historical record is much longer than the archived forecast record. The forecast-assisted supervised model can only use the period for which archived Open-Meteo forecasts overlap with NASA POWER target values. Do not fabricate archived forecasts for older years. The longer NASA record can still support climatology, exploratory analysis and historical visualisations.

## 9. Data Collection and Preprocessing

1. Choose and document representative coordinates for each of the four regions. Because all four regions are coastal (and Mahe is a very small enclave), check that each point falls in a land-dominated grid cell for both Open-Meteo and NASA POWER; move the point slightly inland if necessary.
2. Query NASA POWER for the selected daily solar-radiation target and optional meteorological variables.
3. Query Open-Meteo Previous Runs for archived forecasts at the same requested coordinates and for lead days 1, 2 and 3.
4. Confirm that the selected model and variables are available for all locations and lead times.
5. Aggregate hourly forecast variables to daily features consistently. For example, hourly mean shortwave-radiation values in W/m² can be summed over complete one-hour intervals and divided by 1000 to obtain kWh/m²/day. Alternatively, use a documented daily sum variable if it is available from the selected endpoint.
6. Align date definitions and time standards. Request NASA POWER daily data with `time-standard=LST` (local solar time), and group Open-Meteo hourly forecasts (fetched in UTC) into India Standard Time calendar days. Use this convention consistently for training and live inference. *(Decision from the archive audit: NASA's UTC daily series lagged about 10 weeks behind real time, compared with about 1 week for LST. Both day boundaries fall at night in India, and the LST and UTC daily totals differed by at most 0.001 kWh/m²/day on average.)*
7. Match archived forecasts to NASA target values using target date and region/location.
8. Inspect missing values, duplicate records, invalid values and incomplete days.
9. Fit imputers, encoders, scalers and any feature-selection steps using the training partition only.
10. Save processed datasets, data dictionaries and the retrieval metadata.

## 10. Training, Validation and Evaluation Strategy

### 10.1 Chronological splitting

Use time-based splits rather than random row splitting. For example, an initial experiment might use an earlier period for training, a later period for validation and the most recent complete period for testing. The exact cut-off dates must be selected after auditing the actual archive coverage and label availability.

All lead-time rows for the same target date must be placed in the same split. This prevents the same NASA target date from appearing in both training and testing through different lead-time rows.

Where archive coverage allows, the test period should span at least 12 consecutive months so that every region's monsoon season is represented in the held-out data. Otherwise, a region could be evaluated without its most difficult season.

### 10.2 Evaluation metrics

- **MAE (Mean Absolute Error):** average absolute prediction error, in kWh/m²/day.
- **RMSE (Root Mean Squared Error):** gives greater weight to larger errors, in kWh/m²/day.
- **R² (Coefficient of Determination):** indicates how much target variation is explained by the model.

Report metrics:
- overall;
- separately for lead days 1, 2 and 3; and
- separately for Puducherry, Karaikal, Mahe and Yanam, while clearly noting small sample sizes where applicable.

### 10.3 Baseline comparison

Compare the trained model against:
1. the raw Open-Meteo forecast shortwave-radiation estimate converted to kWh/m²/day; and
2. a simple historical monthly or seasonal average.

The ML model should only be described as improving the forecast if it performs better than the relevant baseline on the held-out chronological test period. If it does not, report that result honestly and retain the raw forecast as a reference.

### 10.4 Regional modelling ablation

To test whether accounting for region improves accuracy, compare:
1. **Pooled, location-blind model** – all regions, no region or region-specific climatology features;
2. **Pooled, region-aware model** – the primary approach (Section 7.6); and
3. **Per-region models** – one model trained separately for each region.

Report MAE, RMSE and R² for each variant separately for every region, alongside the raw Open-Meteo and climatology baselines. Select the regional strategy based on validation performance, and confirm it on the test period.

**Result (Module 6):**
- Region-aware and location-blind models scored within ±0.02 kWh/m²/day of each other on validation. Per-region models were consistently worst.
- The selection rule chose a **location-blind Random Forest** (direct target) by a small margin. The forecast inputs already differ by region, so explicit region features added little with one training year per season.
- The final model scored on the held-out test year (Nov 2025 – Nov 2026):

| | Test MAE (kWh/m²/day) | Test R² |
|---|---|---|
| Final model (location-blind Random Forest) | **0.595** | **0.60** |
| Raw ECMWF forecast | 0.711 | 0.44 |
| Bias-corrected forecast | 0.624 | – |

- The improvement over the raw forecast holds for every lead (15–17 %) and every region (14–19 %).
- The regional comparison should be repeated as more archive years accumulate.

## 11. Live Prediction Workflow

1. The user opens the application and selects one of the four regions on the interactive map.
2. The application retrieves the selected region's configured representative coordinates.
3. The backend requests the next three days of weather forecasts from the Open-Meteo Forecast API.
4. The response is transformed into the exact feature format used during model training.
5. The backend creates one prediction input for each lead day (1, 2 and 3).
6. The saved model generates daily solar-radiation estimates for the selected location.
7. The dashboard displays the three-day estimates, units, forecast issue/retrieval time and relevant context.
8. The interface may show the raw Open-Meteo radiation forecast alongside the ML-adjusted estimate so users can understand the model's contribution.
9. Prediction records and API/model metadata can be stored for reproducibility and later evaluation.

The application should handle API failures, missing variables, incomplete forecast days and stale model files gracefully. It should show an informative message rather than silently presenting old values as new forecasts.

## 12. Application Modules

### Module 1: Geographic Configuration and Map Management

**Purpose:** Define the four regions and connect map selections to the correct coordinates and administrative boundaries.

**Tasks:**
- maintain a configuration file containing region identifiers, names and representative coordinates;
- obtain and validate GeoJSON boundaries for the UT regions and available subdivisions/taluks;
- build the region-selection and drill-down map;
- connect each selectable polygon to its region or administrative identifier; and
- document cases where multiple polygons share the same source grid cell.

**Output:** Geographic configuration, validated GeoJSON files and interactive map layer.

**Implementation:** `scripts/prepare_geo.py` builds the two map layers in `data/geo/`, and `src/geospatial/map_data.py` draws the map.
- **Source:** geoBoundaries gbOpen IND ADM3, which gives sub-districts (taluks) from the Local Government Directory. Licence ODbL 1.0, with attribution shown on the map.
- **Taluks:** the 8 official taluks are used: Puducherry, Ozhukarai, Villianur and Bahour; Karaikal and Thirunallar; Mahe; Yanam.
- **Region outlines:** built by merging each region's taluks. The ADM2 district layer is not used, because it puts Thirunallar inside its "Puducherry" district.
- **Checks:** every representative point lies inside its region.
- **Shared grid cells:** each taluk records which other taluks share its region's grid cell, since all of them use the region's forecast and NASA values.
- **Known gap:** the source outline for Yanam covers about 13 km², against an officially cited 30 km². This affects the map only, not the predictions.

### Module 2: Historical Data Collection – NASA POWER

**Purpose:** Retrieve historical daily solar-radiation reference data.

**Tasks:**
- call the NASA POWER Daily Point API for each representative location;
- retrieve `ALLSKY_SFC_SW_DWN` and selected optional weather variables;
- parse API responses and standardise dates, units and column names;
- store raw responses and cleaned CSV files; and
- record coordinates, request dates and retrieval metadata.

**Output:** `nasa_solar_history.csv` and reproducible data-collection scripts.

### Module 3: Archived Forecast Collection – Open-Meteo

**Purpose:** Obtain historical forecast inputs for supervised learning.

**Tasks:**
- query the Open-Meteo Previous Runs API;
- retrieve the selected variables for forecast lead days 1, 2 and 3;
- record issue time, target date, lead time and forecast model;
- aggregate hourly variables into daily features;
- check coverage and missingness by region, date, variable and lead; and
- save the archived forecast records.

**Output:** `archived_forecasts.csv` and archive-coverage report.

### Module 4: Data Integration and Preprocessing

**Purpose:** Build a reliable model-ready dataset from the two data sources.

**Tasks:**
- align date and location identifiers;
- join archived forecast rows to NASA POWER target values;
- validate coordinate and grid-cell correspondence;
- handle missing values and incomplete forecast days;
- create date, season, region and lead-time features;
- check duplicates, outliers and unit consistency; and
- produce chronological train, validation and test partitions.

**Output:** `solar_model_training.csv`, preprocessing pipeline and data-quality report.

### Module 5: Exploratory Data Analysis and Baseline Development

**Purpose:** Understand solar-radiation patterns and establish reference performance.

**Tasks:**
- analyse monthly and seasonal solar-radiation patterns;
- compare historical patterns among the four regions;
- inspect relationships between forecast variables and the target;
- visualise missingness and data coverage;
- calculate simple monthly/seasonal baselines; and
- compare raw Open-Meteo radiation forecasts with NASA target values.

**Output:** EDA notebook, charts and baseline metrics.

### Module 6: Machine Learning Model Development

**Purpose:** Train, compare and select the solar-radiation prediction model.

**Tasks:**
- establish a Linear Regression baseline;
- train Random Forest and XGBoost regressors;
- tune selected model parameters using training/validation data only;
- evaluate overall and per-lead/per-region performance;
- run the regional modelling ablation (location-blind vs region-aware vs per-region models);
- compare against raw Open-Meteo and climatological baselines; and
- save the chosen model and preprocessing pipeline.

**Output:** Trained model, evaluation results, plots and model metadata.

**Implementation:** there are two saved models.
- **Evaluated model** (`models/evaluated/`): fitted on train + validation by `notebooks/04_model_experiments.ipynb`. Its test-year scores are the reported accuracy.
- **Deployed model** (`models/solar_model.joblib`): the same configuration refitted on all labelled data by `scripts/retrain_production.py`. It has no held-out test of its own, so its accuracy is quoted from the evaluated model.

Periodic retraining (§17) means running `scripts/collect_data.py`, then `scripts/retrain_production.py`.

### Module 7: Live Forecast and Prediction Engine

**Purpose:** Generate three-day predictions from current weather forecasts.

**Tasks:**
- request current forecast data from Open-Meteo;
- validate the response and forecast freshness;
- transform live data using the saved preprocessing pipeline;
- create lead-day 1, 2 and 3 input rows;
- generate predictions using the saved model;
- retain the raw forecast as a comparison; and
- store prediction timestamps and model/API metadata.

**Output:** Three-day solar-radiation predictions for the selected region.

**Implementation:** `src/services/live_forecast.py` and `src/models/predict.py` (`SolarPredictor`); command-line use via `scripts/predict_live.py`. The engine:
- reuses the training transformations unchanged;
- checks that the grid cells match the training grid cells;
- warns when the forecast run is older than 18 h or the model's training data is older than 120 days;
- flags live inputs that are more than 3 standard deviations from the training inputs for the same region and season (±30 days). Drifted inputs the model does not use, such as cloud cover, are monitored but do not mark a prediction as doubtful;
- marks incomplete forecast days as unavailable instead of predicting;
- appends every prediction and its metadata to `data/predictions/prediction_log.csv`.

### Module 8: Interactive Map and Regional Dashboard

**Purpose:** Present model outputs in a clear, location-oriented interface.

**Tasks:**
- display the four UT regions on an interactive map;
- support region and available subdivision/taluk selection;
- show the next three days of predicted solar radiation;
- display a project-defined low/moderate/high availability category;
- plot historical trends and monthly patterns;
- compare regions and forecast lead times; and
- show data-source, update-time and spatial-resolution disclaimers.

**Output:** Interactive web application and regional dashboard.

**Implementation:** `src/dashboard/app.py` (Streamlit, with a Leaflet map through `streamlit-folium` and Plotly charts). Run it with `.venv/Scripts/streamlit run src/dashboard/app.py`.
- **Map:** an overview with labelled markers for the four regions, coloured by tomorrow's category. Clicking a marker or using the dropdown opens a region view that shows its taluks, and those can be clicked too.
- **Forecast cards:** three day cards, each with the ML estimate, its category, and the raw ECMWF forecast. When a taluk is selected, a note explains that all taluks of a region share its values.
- **Tabs:** region comparison, historical patterns, forecast accuracy on the test year (by lead time and by region), and data sources with disclaimers.
- **Failures:** if the API fails, the dashboard shows an error instead of out-of-date values. Warnings from the engine, such as stale runs or unusual inputs, are displayed.

### Module 9: Rooftop Solar-Energy Estimator

**Purpose:** Convert a radiation estimate into a simple, indicative daily energy estimate for a user-entered solar-system capacity.

A simplified estimate is:

`Estimated daily energy (kWh) = Solar radiation (kWh/m²/day) × System capacity (kW) × Performance ratio`

The performance ratio is an assumed factor representing system losses and should be configurable and clearly explained. This is a simplified educational estimate, not a full photovoltaic-system simulation. Actual output also depends on panel efficiency, tilt, orientation, shading, temperature, inverter performance and installation conditions.

**Tasks:**
- accept an assumed rooftop system capacity in kW;
- allow the user to select or view the performance-ratio assumption;
- calculate indicative daily energy for each forecast day; and
- clearly label results as estimates rather than guaranteed generation.

**Output:** Rooftop daily-energy estimate integrated into the dashboard.

**Implementation:** the calculation is `src/services/rooftop.py`, and the calculator appears in the dashboard's region view, below the three day cards. Settings are in the `rooftop` section of `config/pipeline.json`.
- **Formula:** kWp ratings are defined at 1 kW/m² of sunlight, so daily radiation in kWh/m²/day equals peak sun hours. The estimate is therefore the ML radiation estimate × capacity × performance ratio.
- **Inputs:** capacity defaults to 3 kW (allowed range 0.5–100 kW). The performance ratio defaults to 0.75 on a slider from 0.60 to 0.90, with an explanation of the losses it represents.
- **Caveats shown with the result:** it is labelled as an indicative estimate, based on horizontal-surface radiation for the region, and not a measurement at the user's roof.
- **Edge cases:** days without a valid prediction show "—", and invalid inputs are rejected.
- **Tests:** `tests/test_calculator.py`.

### Module 10: Testing, Documentation and Deployment

**Purpose:** Verify the complete application and prepare it for demonstration.

**Tasks:**
- test API parsing, date alignment and unit conversion;
- test preprocessing consistency between training and inference;
- test predictions for all four regions and all three lead times;
- test map selection, dashboard updates and calculator calculations;
- test API timeouts, missing data and stale forecasts;
- document setup, dependencies, API attribution and limitations; and
- deploy the application locally or to an appropriate academic hosting platform.

**Output:** Tested application, README, technical report and demonstration.

**Status (Nov 2026):**
- **Tests:** 42 offline tests in `tests/`, covering every item in the task list above. The dashboard tests use Streamlit's AppTest with a synthetic forecast.
- **Documents:** `README.md` (setup, running, pipeline steps, attribution) and `doc/technical_report.md`.
- **Deployment:** still to do.

## 13. Suggested Technology Stack

| Area | Suggested tools |
|---|---|
| Programming language | Python |
| Data collection | `requests` |
| Data processing | Pandas, NumPy |
| Visualisation and EDA | Matplotlib, Plotly |
| Machine learning | scikit-learn, XGBoost |
| Geospatial processing | GeoPandas, Shapely |
| Interactive map | Leaflet.js with GeoJSON, or Folium/Streamlit components |
| Web application | Streamlit |
| Model persistence | Joblib |
| Development | Jupyter Notebook, VS Code |
| Version control | Git and GitHub |

A practical implementation can use Streamlit for the dashboard and embed a Leaflet-based map for reliable polygon selection and drill-down behaviour.

## 14. Project Folder Structure

```text
pudhuvai-solarscope/
│
├── config/                 locations.json, pipeline.json (all pipeline decisions)
├── data/
│   ├── raw/                cached API responses and boundary downloads (git-ignored)
│   ├── processed/          nasa_solar_history.csv, archived_forecasts.csv,
│   │                       solar_model_training.csv, climatology.csv
│   ├── geo/                puducherry_regions.geojson, administrative_subdivisions.geojson
│   └── predictions/        live prediction log (git-ignored)
├── doc/                    project description, technical report
├── models/                 solar_model.joblib (deployed pipeline incl. preprocessing),
│   │                       model_metadata.json, training_reference.csv
│   └── evaluated/          model whose test scores are reported
├── notebooks/              02_preprocessing, 03_eda_baselines, 04_model_experiments
├── results/                archive_audit/, preprocessing/, eda/, baselines/, models/,
│                           metrics.json, prediction_comparison.png
├── scripts/                audit_archive_coverage, collect_data, retrain_production,
│                           predict_live, prepare_geo
├── src/
│   ├── data_collection/    nasa_power.py, openmeteo_archive.py
│   ├── preprocessing/      clean_data.py, build_training_data.py
│   ├── models/             train.py, evaluate.py, predict.py
│   ├── services/           live_forecast.py, rooftop.py
│   ├── geospatial/         map_data.py
│   ├── dashboard/          app.py
│   └── viz.py
├── tests/                  test_data_processing, test_predictions, test_geo,
│                           test_calculator, test_dashboard
├── requirements.txt
└── README.md
```

## 15. Expected System Output

For a selected region, the application should display:

- predicted daily solar radiation for forecast lead days 1, 2 and 3, in kWh/m²/day;
- the corresponding raw Open-Meteo radiation forecast, where available;
- a simple low/moderate/high availability label based on thresholds defined and documented by the project;
- the forecast issue or retrieval time;
- historical monthly or seasonal solar-radiation charts;
- regional comparisons; and
- an indicative rooftop daily-energy estimate for a user-entered system capacity.

The category thresholds should be selected using domain references or a documented project rule. They should not be presented as official government classifications unless an authoritative standard is identified.

**Project rule (implemented in `config/pipeline.json`):**

| Category | Daily radiation (kWh/m²/day) |
|---|---|
| Low | below 5.0 |
| Moderate | 5.0 to under 6.0 |
| High | 6.0 and above |

The cut-offs are the lower and upper thirds (4.95 and 6.00) of NASA daily `ALLSKY_SFC_SW_DWN` values for 2001–2023, pooled over the four regions and rounded. One rule is used for all regions so they can be compared directly.

## 16. Limitations and Risk Management

1. **Coarse spatial resolution:** NASA POWER grid cells may cover multiple nearby subdivisions. The application must disclose this and must not imply street-level accuracy.
2. **Forecast archive coverage:** Archived forecast availability is shorter than NASA POWER's historical record. Training coverage is limited to the verified overlap period.
3. **Forecast-model consistency:** Archived and live forecasts may differ if the underlying weather model changes. Pin or record the model source and monitor changes.
4. **NASA target uncertainty:** NASA POWER is a gridded estimate and not a local ground-truth solar measurement.
5. **API availability:** Network failures, rate limits or API changes may interrupt data collection or live predictions. Implement retries, caching and clear error messages.
6. **Daily aggregation and time zones:** Inconsistent UTC/local-time aggregation can misalign forecast inputs and target labels. Use a consistent time standard throughout the data pipeline.
7. **Model usefulness:** ML correction may not outperform the raw Open-Meteo radiation forecast. This must be tested and reported honestly.
8. **Rooftop estimate assumptions:** The calculator is indicative and cannot account for every installation-specific factor.
9. **Regional climate differences:** The regions have different monsoon timing, and the short forecast archive contains only a few monsoon cycles per region. Region-aware features reduce, but cannot fully remove, the risk of poor performance in a region's unusual seasons. Per-region metrics must be reported.
10. **Coastal representative points:** Coastal coordinates may fall in grid cells dominated by ocean, which can distort both forecasts and NASA estimates. Representative points must be checked and documented.
11. **Forecast input drift (observed):** archived ECMWF cloud cover changed behaviour from late 2025, and the raw forecast's radiation bias grew from +0.26 to +0.45 kWh/m²/day between 2024 and 2026. Live inputs must be checked against the training distribution, and the model must be retrained periodically.

## 17. Future Enhancements

- Integrate higher-resolution satellite or reanalysis products, such as ERA5-Land or other suitable sources, after checking their resolution, variables and licensing.
- Incorporate local pyranometer measurements or verified rooftop generation data for stronger local validation.
- Add uncertainty intervals or prediction ranges.
- Extend the forecast horizon if data coverage and model performance support it.
- Add downloadable regional reports and historical comparisons.
- Include a more detailed photovoltaic yield model with panel orientation, tilt, shading and system specifications.
- Retrain and monitor the model periodically as new archived forecasts and verified NASA POWER labels become available.

## 18. Final Project Scope

The core deliverable is a working application that:

1. covers Puducherry, Karaikal, Mahe and Yanam;
2. collects NASA POWER historical daily solar-radiation data;
3. pairs NASA target values with archived Open-Meteo forecasts to construct forecast-assisted training examples;
4. trains and evaluates regression models using chronological splits;
5. retrieves live Open-Meteo forecasts and predicts solar radiation for the next three days;
6. presents the results through an interactive administrative map and regional dashboard; and
7. provides a clearly labelled, simplified rooftop solar-energy estimator.

The project succeeds when it provides reproducible data processing, a fair comparison against the raw weather-forecast baseline, transparent treatment of spatial and data limitations, and a usable map-based interface. Its primary prediction output is **daily solar radiation**, not a guaranteed amount of electricity generated by a particular rooftop installation.
