# EV Price AI - Project Technical Overview

## Architecture & System Design

EV Price AI is structured as a decoupled ML inference application with a Streamlit presentation layer and a scikit-learn pipeline backend.

```
+-------------------------------------------------------+
|                 Streamlit User Interface              |
|   (Hero Section, Input Form, Result Cards, Visuals)   |
+-------------------------------------------------------+
                           |
                           v  Input DataFrame (21 features)
+-------------------------------------------------------+
|             Joblib Model Loader & Validator           |
|            (ev_price_prediction_model.pkl)            |
+-------------------------------------------------------+
                           |
                           v Pipeline execution
+-------------------------------------------------------+
|             Scikit-Learn Pipeline Backend             |
|   - Preprocessing (OneHotEncoder / StandardScaler)    |
|   - Random Forest Regressor Estimator                 |
+-------------------------------------------------------+
                           |
                           v Prediction output ($)
+-------------------------------------------------------+
|              Formatted Result Display UI              |
+-------------------------------------------------------+
```

## Input Feature Schema (All 21 Model Columns)

| Feature Name | Type | Description | Range / Categories |
|---|---|---|---|
| `brand` | Categorical | EV Manufacturer | Text input (e.g. Tesla, Porsche) |
| `model` | Categorical | Model Name | Text input (e.g. Model 3, Taycan) |
| `variant` | Categorical | Trim / Variant | Text input (e.g. Long Range) |
| `battery_capacity_kwh` | Numerical | Battery Pack Energy | 5.0 to 250.0 kWh |
| `range_miles` | Numerical | EPA Driving Range | 10.0 to 1000.0 miles |
| `horsepower` | Numerical | Motor Power Output | 20.0 to 2000.0 HP |
| `body_type` | Categorical | Vehicle Design Style | Sedan, SUV, Hatchback, Coupe, MPV, Truck, Van |
| `weight_kg` | Numerical | Curb Weight | 500.0 to 5000.0 kg |
| `charging_speed_kw` | Numerical | DC Fast Charging Speed | 10.0 to 500.0 kW |
| `autopilot_level` | Categorical / Int | Autonomous Level | 0, 1, 2, 3 |
| `year` | Categorical / Int | Model Year | 2020 to 2026 |
| `top_speed_mph` | Numerical | Maximum Speed | 50.0 to 300.0 mph |
| `acceleration_0_60_mph` | Numerical | 0-60 mph Time | 1.0 to 15.0 sec |
| `drive_type` | Categorical | Drivetrain Layout | AWD, FWD, RWD |
| `seating_capacity` | Categorical / Int | Seat Count | 2, 4, 5, 7 |
| `warranty_years` | Categorical / Int | Warranty Period | 3, 4, 5, 6, 7, 8 years |
| `market_segment` | Categorical | Market Classification | Budget, Mid-range, Premium, Luxury |
| `cargo_volume_cubic_ft` | Numerical | Cargo Storage Space | 5.0 to 150.0 cu ft |
| `torque_nm` | Numerical | Motor Torque | 50.0 to 2000.0 Nm |
| `country_of_origin` | Categorical | Manufacturing Location | US, Germany, China, Japan, South Korea |
| `safety_rating` | Categorical / Int | Safety Star Rating | 3, 4, 5 stars |

## Model Compatibility & Safety Features

1. **Exact 21-Feature Ordering**: Features are passed to `pipeline.predict()` inside a pandas DataFrame matching the exact 21 feature columns expected by the trained preprocessing pipeline.
2. **Type Safety**: Input parameters are properly typed (floats for continuous specs, ints for ordinal specs, strings for categorical specs).
3. **Graceful Fallbacks**: Includes candidate path resolution to locate `ev_price_prediction_model.pkl` across relative, script-relative, and local installation paths.
