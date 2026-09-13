#  EV Price AI — Electric Vehicle Price Predictor

An AI-powered web application that predicts the estimated price of an Electric Vehicle based on its specifications using Machine Learning.

The project combines a trained **Random Forest Regression model** with a **Streamlit web interface** to provide an interactive EV price prediction experience.

---

##  Project Overview

Electric vehicle prices depend on multiple factors such as battery capacity, driving range, horsepower, vehicle weight, charging speed, performance, safety, and market characteristics.

This project uses machine learning to analyze these vehicle specifications and estimate the expected EV price.

### Project Workflow

Vehicle Specifications
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Random Forest Regression
        ↓
Predicted EV Price
        ↓
Streamlit Web Application

---

## ✨ Features

- 🔋 Battery capacity based prediction
- 🚗 Vehicle specification analysis
- ⚡ EV performance parameters
- 📊 Machine Learning based price estimation
- 🖥️ Interactive Streamlit web application
- 📱 Responsive and modern user interface
- 🤖 Random Forest Regression model
- 📈 Data analysis and model training notebook

---

## 🧠 Machine Learning

### Model Used

**Random Forest Regressor**

The model is trained using EV-related vehicle specifications to estimate the price of an electric vehicle.

### Input Features

The model uses vehicle parameters including:

- Brand
- Model
- Variant
- Battery Capacity
- Range
- Horsepower
- Body Type
- Weight
- Charging Speed
- Autopilot Level
- Year
- Top Speed
- Acceleration
- Drive Type
- Seating Capacity
- Warranty
- Market Segment
- Cargo Volume
- Torque
- Country of Origin
- Safety Rating

---

## 🗂️ Project Structure

```text
EV-Price-Predictor/
│
├── app.py
├── EV_Price_Prediction.ipynb
├── ev_market_2026.csv
├── requirements.txt
├── README.md
│
├── docs/
│   └── project_overview.md
│
└── assets/
    └── screenshots/
