import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Predictive Analytics Dashboard", layout="wide")

st.title("Predictive Analytics Using Historical Data")
st.write("Forecast future trends and perform data-driven analysis using machine learning.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Upload dataset
st.sidebar.subheader("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type=["csv"])

@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("data/historical_sales.csv")
    except FileNotFoundError:
        # Create dummy data if not found
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        trend = np.linspace(10, 50, len(dates))
        seasonality = 10 * np.sin(np.arange(len(dates)) * (2 * np.pi / 365))
        noise = np.random.normal(0, 5, len(dates))
        sales = trend + seasonality + noise
        return pd.DataFrame({'Date': dates, 'Sales': sales})

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default historical sales dataset. You can upload your own CSV file in the sidebar.")
    df = load_default_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocessing
st.subheader("Data Preprocessing & Cleaning")
st.write("Automated cleaning and preparing data for predictive modeling...")

# 1. Handle Duplicates
initial_shape = df.shape
df = df.drop_duplicates()
if initial_shape[0] != df.shape[0]:
    st.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
else:
    st.success("No duplicate rows found.")

# 2. Handle Missing Values
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    st.warning(f"Found {missing_data.sum()} missing values. Handling them automatically...")
    # Fill numeric columns with median
    num_cols_missing = df.select_dtypes(include=[np.number]).columns
    for col in num_cols_missing:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    cat_cols_missing = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols_missing:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    st.success("Missing values imputed successfully (Numeric: Median, Categorical: Mode).")
else:
    st.success("No missing values found.")

# Automated Data Quality Report
with st.expander("View Automated Data Quality Report"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("Data Summary:")
    st.dataframe(df.describe())
    
    st.write("Data Types:")
    st.dataframe(pd.DataFrame(df.dtypes.astype(str), columns=['Data Type']))

# Extract numeric columns for the target variable
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Please upload a dataset with at least one numeric column to predict.")
    st.stop()

# Select columns
date_col = st.sidebar.selectbox("Select Date Column", df.columns, index=0)
target_col = st.sidebar.selectbox("Select Target Variable to Predict", numeric_cols, index=0)

# Process Date Column
try:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    st.success(f"Successfully parsed '{date_col}' as Date.")
except Exception as e:
    st.error(f"Error parsing date column: {e}. Please ensure it's a valid date format.")
    st.stop()

# Feature Engineering for Time Series
df['Year'] = df[date_col].dt.year
df['Month'] = df[date_col].dt.month
df['Day'] = df[date_col].dt.day
df['DayOfWeek'] = df[date_col].dt.dayofweek
df['DayOfYear'] = df[date_col].dt.dayofyear
df['Days_Since_Start'] = (df[date_col] - df[date_col].min()).dt.days

st.write("Generated temporal features: Year, Month, Day, DayOfWeek, DayOfYear, Days_Since_Start")
st.dataframe(df.head())

# Visualization of Historical Data
st.subheader(f"Historical Trend: {target_col} over Time")
fig_hist = px.line(df, x=date_col, y=target_col, title=f"Historical {target_col}")
st.plotly_chart(fig_hist, use_container_width=True)

# Modeling
st.sidebar.subheader("Modeling")
model_type = st.sidebar.selectbox("Select Predictive Model", ["Linear Regression", "Random Forest Regressor"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days into future)", min_value=7, max_value=365, value=30)

features = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Days_Since_Start']
X = df[features]
y = df[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

if model_type == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Evaluation
y_pred_test = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

st.subheader("Model Evaluation")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R-Squared (R²)", f"{r2:.2f}")

# Visualize Test Predictions
fig_eval = go.Figure()
fig_eval.add_trace(go.Scatter(x=df[date_col].iloc[X_train.index], y=y_train, mode='lines', name='Training Data'))
fig_eval.add_trace(go.Scatter(x=df[date_col].iloc[X_test.index], y=y_test, mode='lines', name='Actual Test Data'))
fig_eval.add_trace(go.Scatter(x=df[date_col].iloc[X_test.index], y=y_pred_test, mode='lines', name='Predicted Test Data', line=dict(dash='dash')))
fig_eval.update_layout(title="Model Fit on Test Data", xaxis_title="Date", yaxis_title=target_col)
st.plotly_chart(fig_eval, use_container_width=True)

# Forecasting Future Trends
st.subheader("Future Forecasting")
last_date = df[date_col].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

future_df = pd.DataFrame({date_col: future_dates})
future_df['Year'] = future_df[date_col].dt.year
future_df['Month'] = future_df[date_col].dt.month
future_df['Day'] = future_df[date_col].dt.day
future_df['DayOfWeek'] = future_df[date_col].dt.dayofweek
future_df['DayOfYear'] = future_df[date_col].dt.dayofyear
future_df['Days_Since_Start'] = (future_df[date_col] - df[date_col].min()).dt.days

X_future = future_df[features]
future_df['Predicted_' + target_col] = model.predict(X_future)

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df[date_col], y=df[target_col], mode='lines', name='Historical Data'))
fig_forecast.add_trace(go.Scatter(x=future_df[date_col], y=future_df['Predicted_' + target_col], mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
fig_forecast.update_layout(title=f"Forecasted {target_col} for next {forecast_horizon} days", xaxis_title="Date", yaxis_title=target_col)
st.plotly_chart(fig_forecast, use_container_width=True)

st.success("Forecasting completed! You can experiment with different models and forecast horizons in the sidebar.")
