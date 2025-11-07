import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
import os
import json

# Disable warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Supabase Configuration - Hardcoded
SUPABASE_URL = "https://fjfmgndbiespptmsnrff.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"

# Page configuration
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🌍 Air Quality Prediction Dashboard")
st.markdown("Predict future air quality metrics using machine learning models")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None

# Fetch data from Supabase
@st.cache_data(ttl=300)
def fetch_data():
    try:
        supabase = init_supabase()
        if supabase is None:
            return None
        
        response = supabase.table('airquality').select('*').order('created_at', desc=False).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at').reset_index(drop=True)
            return df
        else:
            st.warning("No data found in the database")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Feature engineering
def create_time_features(df):
    if df is None or len(df) == 0:
        return df
        
    df = df.copy()
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['month'] = df['created_at'].dt.month
    
    # Create lag features
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    for metric in metrics:
        if metric in df.columns:
            df[f'{metric}_lag1'] = df[metric].shift(1)
            df[f'{metric}_lag2'] = df[metric].shift(2)
    
    df = df.dropna()
    return df

# Train models
@st.cache_resource
def train_models(df):
    if df is None or len(df) < 10:
        return None
    
    df_features = create_time_features(df)
    
    if df_features is None or len(df_features) < 5:
        st.error("Not enough data for training")
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    models = {'rf': {}, 'xgb': {}}
    performance = {'rf': {}, 'xgb': {}}
    
    base_features = ['hour', 'day_of_week', 'month']
    
    for metric in metrics:
        if metric not in df_features.columns:
            continue
            
        lag_features = [col for col in df_features.columns if metric in col and col != metric]
        features = base_features + lag_features
        features = [f for f in features if f in df_features.columns]
        
        if len(features) == 0:
            continue
            
        X = df_features[features]
        y = df_features[metric]
        
        if len(X) < 10:
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        if len(X_train) == 0:
            continue
        
        # Train Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)
            
            models['rf'][metric] = {
                'model': rf_model,
                'features': features
            }
            
            performance['rf'][metric] = {
                'MAE': rf_mae,
                'RMSE': rf_rmse,
                'R2': rf_r2
            }
        except Exception as e:
            st.warning(f"RF failed for {metric}: {str(e)}")
    
    return models, performance, df_features

# Make predictions
def predict_future(models, df_features, hours_ahead=1, model_type='rf'):
    if models is None or df_features is None or len(df_features) == 0:
        return None
    
    if model_type not in models:
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    model_dict = models[model_type]
    
    current_data = df_features.iloc[-1].copy()
    last_timestamp = current_data['created_at']
    
    # Predict for target hour
    step_timestamp = last_timestamp + timedelta(hours=hours_ahead)
    
    time_features = {
        'hour': step_timestamp.hour,
        'day_of_week': step_timestamp.dayofweek,
        'month': step_timestamp.month
    }
    
    predictions = {}
    for metric in metrics:
        if metric not in model_dict:
            predictions[metric] = 0
            continue
            
        model_data = model_dict[metric]
        model = model_data['model']
        features = model_data['features']
        
        X_pred = []
        for feature in features:
            if feature in time_features:
                X_pred.append(time_features[feature])
            elif feature in current_data:
                X_pred.append(current_data[feature])
            else:
                X_pred.append(0)
        
        try:
            pred_value = model.predict([X_pred])[0]
            predictions[metric] = pred_value
        except Exception as e:
            predictions[metric] = current_data.get(metric, 0)
    
    predictions['timestamp'] = step_timestamp
    return predictions

def predict_all_models(models, df_features, hours_ahead=1):
    predictions = {}
    for model_type in ['rf', 'xgb']:
        if model_type in models and models[model_type]:
            predictions[model_type] = predict_future(models, df_features, hours_ahead, model_type)
    return predictions

# Create simple plot
def create_prediction_plot(predictions, metric_name):
    fig = go.Figure()
    
    if predictions:
        model_names = {
            'rf': 'Random Forest',
            'xgb': 'XGBoost'
        }
        
        for model_type in ['rf', 'xgb']:
            if model_type in predictions and predictions[model_type]:
                pred_data = predictions[model_type]
                if metric_name.lower() in pred_data:
                    fig.add_trace(go.Indicator(
                        mode = "number+delta",
                        value = pred_data[metric_name.lower()],
                        title = {"text": f"{model_names[model_type]}"},
                        number = {'suffix': get_metric_suffix(metric_name)},
                        domain = {'x': [0, 1], 'y': [0, 1]}
                    ))
    
    fig.update_layout(
        title=f'{metric_name} Predictions',
        height=200
    )
    return fig

def get_metric_suffix(metric_name):
    suffixes = {
        'Temperature': '°C',
        'Humidity': '%',
        'CO2': ' ppm',
        'CO': ' ppm',
        'PM2.5': ' µg/m³',
        'PM10': ' µg/m³'
    }
    return suffixes.get(metric_name, '')

# Main app
def main():
    st.sidebar.header("⚙️ Settings")
    
    # Fetch data
    with st.spinner("Fetching data from Supabase..."):
        df = fetch_data()
    
    if df is None or len(df) == 0:
        st.error("""
        Unable to load data from database. 
        
        **Possible reasons:**
        - No data in the 'airquality' table
        - Network connection issues
        - Database structure mismatch
        
        Please check your Supabase database and try again.
        """)
        return
    
    st.sidebar.success(f"✅ Loaded {len(df)} records")
    
    # Check required columns
    required_columns = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    available_columns = [col for col in required_columns if col in df.columns]
    
    if len(available_columns) == 0:
        st.error("No required columns found in the data")
        return
    
    # Train models
    with st.spinner("Training machine learning models..."):
        result = train_models(df)
        
        if result is None:
            st.error("Unable to train models. Not enough data available.")
            return
        
        models, performance, df_features = result
    
    st.sidebar.success("✅ Models trained successfully")
    
    # Prediction settings
    st.sidebar.subheader("Prediction Settings")
    forecast_hours = st.sidebar.slider(
        "Forecast Hours Ahead",
        min_value=1,
        max_value=12,
        value=6
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔮 Predictions", "📈 Historical Trends"])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Recent Data")
        
        if len(df) > 0:
            latest = df.iloc[-1]
            cols = st.columns(6)
            
            metric_configs = [
                ("Temperature", "temperature", "°C"),
                ("Humidity", "humidity", "%"),
                ("CO2", "co2", "ppm"),
                ("CO", "co", "ppm"),
                ("PM2.5", "pm25", "µg/m³"),
                ("PM10", "pm10", "µg/m³")
            ]
            
            for i, (name, key, unit) in enumerate(metric_configs):
                if key in latest:
                    with cols[i]:
                        st.metric(name, f"{latest[key]:.1f}{unit}")
        
        st.subheader("Data Statistics")
        if available_columns:
            stats_df = df[available_columns].describe().T
            st.dataframe(stats_df, use_container_width=True)
        
        st.subheader("Latest Records")
        display_cols = ['created_at'] + available_columns
        st.dataframe(df[display_cols].tail(20), use_container_width=True)
    
    # Tab 2: Predictions
    with tab2:
        st.header("Future Predictions")
        
        all_predictions = predict_all_models(models, df_features, hours_ahead=forecast_hours)
        
        if all_predictions:
            st.subheader(f"Predictions for {forecast_hours} hours ahead")
            
            # Current values
            if len(df) > 0:
                latest = df.iloc[-1]
                st.write("**Current Values:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Temperature", f"{latest['temperature']:.1f}°C")
                    st.metric("Humidity", f"{latest['humidity']:.1f}%")
                with col2:
                    st.metric("CO2", f"{latest['co2']:.0f} ppm")
                    st.metric("CO", f"{latest['co']:.2f} ppm")
                with col3:
                    st.metric("PM2.5", f"{latest['pm25']:.1f} µg/m³")
                    st.metric("PM10", f"{latest['pm10']:.1f} µg/m³")
            
            # Predictions
            st.write("**Predictions:**")
            metrics_display = ['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10']
            
            for metric in metrics_display:
                fig = create_prediction_plot(all_predictions, metric)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Historical Trends
    with tab3:
        st.header("Historical Data")
        
        available_metrics = [
            ('Temperature', 'temperature', '°C'),
            ('Humidity', 'humidity', '%'),
            ('CO2', 'co2', 'ppm'),
            ('CO', 'co', 'ppm'),
            ('PM2.5', 'pm25', 'µg/m³'),
            ('PM10', 'pm10', 'µg/m³')
        ]
        available_metrics = [(name, key, unit) for name, key, unit in available_metrics if key in df.columns]
        
        if available_metrics:
            selected_metric = st.selectbox(
                "Select Metric",
                options=available_metrics,
                format_func=lambda x: x[0]
            )
            
            metric_name, metric_key, unit = selected_metric
            
            fig = px.line(
                df,
                x='created_at',
                y=metric_key,
                title=f'{metric_name} Over Time',
                labels={'created_at': 'Time', metric_key: f'{metric_name} ({unit})'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[metric_key].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[metric_key].median():.2f}")
            with col3:
                st.metric("Min", f"{df[metric_key].min():.2f}")
            with col4:
                st.metric("Max", f"{df[metric_key].max():.2f}")

if __name__ == "__main__":
    main()
