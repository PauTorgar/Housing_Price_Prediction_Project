# Housing Price Prediction Dashboard

Interactive Streamlit dashboard for housing price prediction using the Kaggle House Prices dataset.

## Overview

This project turns a machine learning workflow into a live dashboard for:

- Exploring housing sale price patterns
- Comparing regression model performance
- Reviewing the most important model signals
- Generating an instant sale price estimate from a custom property profile

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly

## Models

The app trains and compares three regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is selected by validation RMSE.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Then open:

```text
(https://housingpricepredictionproject.streamlit.app/)
```

## Project Files

- `app.py`: Streamlit dashboard
- `train.csv`: training dataset used by the app
- `test.csv`: Kaggle test dataset
- `Housing_Price_Prediction_Portfolio.ipynb`: notebook version of the analysis
