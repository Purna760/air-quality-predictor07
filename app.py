import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Supabase Configuration
SUPABASE_URL = "https://fjfmgndbiespptmsnrff.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"

# Page configuration
st.set_page_config(
    page_title="Air Quality Monitor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Monitoring Dashboard")
st.markdown("Real-time air quality data visualization and analysis")

# Initialize Supabase
@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Supabase connection failed: {str(e)}")
        return None

# Fetch data
@st.cache_data(ttl=300)
def fetch_data():
    try:
        supabase = init_supabase()
        if not supabase:
            return None
            
        response = supabase.table('airquality').select('*').order('created_at', desc=False).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df.sort_values('created_at').reset_index(drop=True)
        return None
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return None

# Simple feature engineering
def add_basic_features(df):
    if df is None or len(df) == 0:
        return df
        
    df = df.copy()
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    return df

# Train simple model
def train_simple_model(df, target_column):
    if df is None or len(df) < 10 or target_column not in df.columns:
        return None
        
    df_features = add_basic_features(df)
    features = ['hour', 'day_of_week']
    
    # Add lag features if we have enough data
    if len(df_features) > 5:
        df_features[f'{target_column}_lag1'] = df_features[target_column].shift(1)
        features.append(f'{target_column}_lag1')
    
    df_features = df_features.dropna()
    
    if len(df_features) < 5:
        return None
        
    X = df_features[features]
    y = df_features[target_column]
    
    if len(X) < 5:
        return None
        
    try:
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
        model.fit(X, y)
        return model, features
    except Exception:
        return None

# Simple prediction
def simple_predict(model, features, last_data, hours_ahead):
    if model is None:
        return None
        
    prediction_time = last_data['created_at'] + timedelta(hours=hours_ahead)
    
    # Prepare features
    feature_values = []
    for feature in features:
        if feature == 'hour':
            feature_values.append(prediction_time.hour)
        elif feature == 'day_of_week':
            feature_values.append(prediction_time.dayofweek)
        elif feature.endswith('_lag1'):
            feature_values.append(last_data.get(feature, last_data.get(feature.replace('_lag1', ''), 0)))
        else:
            feature_values.append(0)
    
    try:
        prediction = model.predict([feature_values])[0]
        return prediction
    except Exception:
        return None

def main():
    # Load data
    with st.spinner("Loading data..."):
        df = fetch_data()
    
    if df is None or len(df) == 0:
        st.error("""
        ❗ No data available
        
        **Please check:**
        1. Your Supabase database has data in the 'airquality' table
        2. The table has columns: created_at, temperature, humidity, co2, co, pm25, pm10
        3. Your internet connection is working
        """)
        return
    
    st.success(f"✅ Loaded {len(df)} records")
    
    # Show basic info
    st.sidebar.header("Dashboard Info")
    st.sidebar.metric("Total Records", len(df))
    if len(df) > 0:
        st.sidebar.metric("Date Range", 
                         f"{df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Data", "📈 Trends", "🔮 Predictions"])
    
    with tab1:
        st.header("Current Air Quality")
        
        if len(df) > 0:
            latest = df.iloc[-1]
            cols = st.columns(6)
            
            metrics = [
                ("🌡️ Temperature", "temperature", "°C", "{:.1f}"),
                ("💧 Humidity", "humidity", "%", "{:.1f}"),
                ("🌫️ CO2", "co2", "ppm", "{:.0f}"),
                ("🔥 CO", "co", "ppm", "{:.2f}"),
                ("🫧 PM2.5", "pm25", "µg/m³", "{:.1f}"),
                ("💨 PM10", "pm10", "µg/m³", "{:.1f}")
            ]
            
            for i, (icon, key, unit, fmt) in enumerate(metrics):
                if key in latest:
                    with cols[i]:
                        st.metric(icon, fmt.format(latest[key]) + unit)
        
        st.subheader("Recent Data")
        display_cols = ['created_at']
        for col in ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']:
            if col in df.columns:
                display_cols.append(col)
        
        st.dataframe(df[display_cols].tail(10), use_container_width=True)
    
    with tab2:
        st.header("Historical Trends")
        
        available_metrics = []
        for name, key in [("Temperature", "temperature"), ("Humidity", "humidity"), 
                         ("CO2", "co2"), ("CO", "co"), ("PM2.5", "pm25"), ("PM10", "pm10")]:
            if key in df.columns:
                available_metrics.append((name, key))
        
        if available_metrics:
            selected_metric = st.selectbox("Select metric", available_metrics, format_func=lambda x: x[0])
            metric_name, metric_key = selected_metric
            
            fig = px.line(df, x='created_at', y=metric_key, 
                         title=f'{metric_name} Over Time',
                         labels={'created_at': 'Time', metric_key: metric_name})
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic stats
            if metric_key in df.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average", f"{df[metric_key].mean():.2f}")
                with col2:
                    st.metric("Maximum", f"{df[metric_key].max():.2f}")
                with col3:
                    st.metric("Minimum", f"{df[metric_key].min():.2f}")
                with col4:
                    st.metric("Latest", f"{df[metric_key].iloc[-1]:.2f}")
    
    with tab3:
        st.header("Simple Predictions")
        
        st.info("This shows basic trend-based predictions using Random Forest")
        
        available_targets = []
        for name, key in [("Temperature", "temperature"), ("Humidity", "humidity"), 
                         ("PM2.5", "pm25"), ("PM10", "pm10")]:
            if key in df.columns:
                available_targets.append((name, key))
        
        if available_targets:
            selected_target = st.selectbox("Predict", available_targets, format_func=lambda x: x[0])
            target_name, target_key = selected_target
            
            forecast_hours = st.slider("Hours ahead", 1, 12, 6)
            
            if st.button("Generate Prediction"):
                with st.spinner("Training model..."):
                    model_result = train_simple_model(df, target_key)
                
                if model_result:
                    model, features = model_result
                    latest_data = add_basic_features(df).iloc[-1]
                    
                    prediction = simple_predict(model, features, latest_data, forecast_hours)
                    
                    if prediction is not None:
                        current_value = df[target_key].iloc[-1]
                        
                        st.success(f"**Prediction for {forecast_hours} hours ahead:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current", f"{current_value:.2f}")
                        with col2:
                            st.metric("Predicted", f"{prediction:.2f}", 
                                     delta=f"{(prediction - current_value):.2f}")
                    else:
                        st.warning("Could not generate prediction")
                else:
                    st.warning("Not enough data for reliable predictions")
        else:
            st.warning("No suitable data columns for prediction")

if __name__ == "__main__":
    main()
