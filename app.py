import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
import os
import json
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Supabase Configuration - Hardcoded with correct credentials
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
st.markdown("Predict future air quality metrics (PM2.5, PM10, CO2, CO, Temperature, Humidity) using **Random Forest, XGBoost, and LSTM** models")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection
        response = supabase.table('airquality').select('*', count='exact').limit(1).execute()
        return supabase
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None

# Fetch data from Supabase
@st.cache_data(ttl=300)
def fetch_data():
    """Fetch air quality data from Supabase"""
    try:
        supabase = init_supabase()
        if supabase is None:
            return None
        
        response = supabase.table('airquality').select('*').order('created_at', desc=False).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Sort by timestamp
            df = df.sort_values('created_at').reset_index(drop=True)
            
            return df
        else:
            st.warning("No data found in the database")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Feature engineering for time series
def create_time_features(df):
    """Create time-based features for better predictions"""
    if df is None or len(df) == 0:
        return df
        
    df = df.copy()
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['day_of_month'] = df['created_at'].dt.day
    df['month'] = df['created_at'].dt.month
    
    # Create lag features for each metric
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    for metric in metrics:
        if metric in df.columns:
            df[f'{metric}_lag1'] = df[metric].shift(1)
            df[f'{metric}_lag2'] = df[metric].shift(2)
            df[f'{metric}_lag3'] = df[metric].shift(3)
            df[f'{metric}_rolling_mean_3'] = df[metric].rolling(window=3, min_periods=1).mean()
            df[f'{metric}_rolling_std_3'] = df[metric].rolling(window=3, min_periods=1).std()
    
    # Drop rows with NaN values from lag features
    df = df.dropna()
    
    return df

# Train models for each metric
@st.cache_resource
def train_models(df):
    """Train Random Forest, XGBoost, and LSTM models for each air quality metric"""
    if df is None or len(df) < 10:
        return None
    
    # Create features
    df_features = create_time_features(df)
    
    if df_features is None or len(df_features) < 5:
        st.error("Not enough data for training after feature engineering")
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    models = {'rf': {}, 'xgb': {}, 'lstm': {}}
    performance = {'rf': {}, 'xgb': {}, 'lstm': {}}
    
    # Base features (time features + all lag features)
    base_features = ['hour', 'day_of_week', 'day_of_month', 'month']
    
    for metric in metrics:
        if metric not in df_features.columns:
            continue
            
        # Features for this metric
        lag_features = [col for col in df_features.columns if metric in col and col != metric]
        features = base_features + lag_features
        
        # Ensure all features exist in dataframe
        features = [f for f in features if f in df_features.columns]
        
        if len(features) == 0:
            continue
            
        X = df_features[features]
        y = df_features[metric]
        
        # Check if we have enough data
        if len(X) < 10:
            continue
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Scale data for LSTM
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
        
        # 1. Train Random Forest model
        try:
            rf_model = RandomForestRegressor(
                n_estimators=50,  # Reduced for faster training
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
                'features': features,
                'scaler_X': None,
                'scaler_y': None
            }
            
            performance['rf'][metric] = {
                'MAE': rf_mae,
                'RMSE': rf_rmse,
                'R2': rf_r2,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        except Exception as e:
            st.warning(f"Failed to train Random Forest for {metric}: {str(e)}")
        
        # 2. Train XGBoost model
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=50,  # Reduced for faster training
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train, verbose=False)
            xgb_pred = xgb_model.predict(X_test)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            xgb_r2 = r2_score(y_test, xgb_pred)
            
            models['xgb'][metric] = {
                'model': xgb_model,
                'features': features,
                'scaler_X': None,
                'scaler_y': None
            }
            
            performance['xgb'][metric] = {
                'MAE': xgb_mae,
                'RMSE': xgb_rmse,
                'R2': xgb_r2,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        except Exception as e:
            st.warning(f"Failed to train XGBoost for {metric}: {str(e)}")
        
        # 3. Train LSTM model (only if we have enough data)
        try:
            if len(X_train) > 20:  # Minimum data for LSTM
                # Reshape for LSTM (samples, timesteps, features)
                X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                
                lstm_model = Sequential([
                    LSTM(32, activation='relu', return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                    Dropout(0.2),
                    LSTM(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                lstm_model.fit(
                    X_train_lstm, y_train_scaled,
                    epochs=30,  # Reduced for faster training
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
                lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).flatten()
                lstm_mae = mean_absolute_error(y_test, lstm_pred)
                lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
                lstm_r2 = r2_score(y_test, lstm_pred)
                
                models['lstm'][metric] = {
                    'model': lstm_model,
                    'features': features,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y
                }
                
                performance['lstm'][metric] = {
                    'MAE': lstm_mae,
                    'RMSE': lstm_rmse,
                    'R2': lstm_r2,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
            else:
                st.warning(f"Not enough data for LSTM training on {metric}")
        except Exception as e:
            st.warning(f"Failed to train LSTM for {metric}: {str(e)}")
    
    return models, performance, df_features

# Make predictions
def predict_future(models, df_features, hours_ahead=1, model_type='rf'):
    """Predict future values for all metrics using iterative multi-step forecasting for a specific model"""
    if models is None or df_features is None or len(df_features) == 0:
        return None
    
    if model_type not in models:
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    model_dict = models[model_type]
    
    # Get the latest data point
    current_data = df_features.iloc[-1].copy()
    last_timestamp = current_data['created_at']
    
    # Initialize predictions dictionary
    step_predictions = {}
    
    # Iteratively predict for each hour
    for step in range(1, hours_ahead + 1):
        # Calculate timestamp for this step
        step_timestamp = last_timestamp + timedelta(hours=step)
        
        # Update time features for this step
        time_features = {
            'hour': step_timestamp.hour,
            'day_of_week': step_timestamp.dayofweek,
            'day_of_month': step_timestamp.day,
            'month': step_timestamp.month
        }
        
        # Predict each metric for this step
        step_predictions = {}
        for metric in metrics:
            if metric not in model_dict:
                continue
                
            model_data = model_dict[metric]
            model = model_data['model']
            features = model_data['features']
            scaler_X = model_data['scaler_X']
            scaler_y = model_data['scaler_y']
            
            # Prepare features for prediction
            X_pred = []
            for feature in features:
                if feature in time_features:
                    X_pred.append(time_features[feature])
                elif feature in current_data:
                    X_pred.append(current_data[feature])
                else:
                    X_pred.append(0)  # Default value if feature missing
            
            # Make prediction based on model type
            try:
                if model_type == 'lstm' and scaler_X is not None and scaler_y is not None:
                    X_pred_array = np.array(X_pred).reshape(1, -1)
                    X_pred_scaled = scaler_X.transform(X_pred_array)
                    X_pred_lstm = X_pred_scaled.reshape((1, 1, X_pred_scaled.shape[1]))
                    pred_scaled = model.predict(X_pred_lstm, verbose=0)
                    pred_value = scaler_y.inverse_transform(pred_scaled).flatten()[0]
                else:
                    pred_value = model.predict([X_pred])[0]
                
                step_predictions[metric] = pred_value
            except Exception as e:
                st.warning(f"Prediction failed for {metric}: {str(e)}")
                step_predictions[metric] = current_data.get(metric, 0)
        
        # Roll forward: Update current_data with predictions for next iteration
        for metric in metrics:
            if metric not in step_predictions:
                continue
                
            # Capture old lag values before overwriting
            old_lag1 = current_data.get(f'{metric}_lag1', step_predictions[metric])
            old_lag2 = current_data.get(f'{metric}_lag2', step_predictions[metric])
            
            # Shift lag values
            current_data[f'{metric}_lag3'] = current_data.get(f'{metric}_lag2', old_lag1)
            current_data[f'{metric}_lag2'] = old_lag1
            current_data[f'{metric}_lag1'] = step_predictions[metric]
            
            # Update rolling statistics using distinct recent values
            recent_values = [
                step_predictions[metric],
                old_lag1,
                old_lag2
            ]
            current_data[f'{metric}_rolling_mean_3'] = np.mean(recent_values)
            current_data[f'{metric}_rolling_std_3'] = np.std(recent_values)
        
        # Update timestamp
        current_data['created_at'] = step_timestamp
    
    # Return the final predictions (for the target hour)
    predictions = {metric: step_predictions.get(metric, 0) for metric in metrics}
    predictions['timestamp'] = last_timestamp + timedelta(hours=hours_ahead)
    
    return predictions

def predict_hourly(models, df_features, hours_ahead=24, model_type='rf'):
    """Predict future values for all metrics hour by hour, returning predictions for EACH hour"""
    if models is None or df_features is None or len(df_features) == 0:
        return None
    
    if model_type not in models:
        return None
        
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    model_dict = models[model_type]
    
    # Get the latest data point
    current_data = df_features.iloc[-1].copy()
    last_timestamp = current_data['created_at']
    
    # Initialize results storage for all hourly predictions
    hourly_predictions = {metric: [] for metric in metrics}
    hourly_timestamps = []
    
    # Iteratively predict for each hour
    for step in range(1, hours_ahead + 1):
        # Calculate timestamp for this step
        step_timestamp = last_timestamp + timedelta(hours=step)
        hourly_timestamps.append(step_timestamp)
        
        # Update time features for this step
        time_features = {
            'hour': step_timestamp.hour,
            'day_of_week': step_timestamp.dayofweek,
            'day_of_month': step_timestamp.day,
            'month': step_timestamp.month
        }
        
        # Predict each metric for this step
        step_predictions = {}
        for metric in metrics:
            if metric not in model_dict:
                hourly_predictions[metric].append(0)
                continue
                
            model_data = model_dict[metric]
            model = model_data['model']
            features = model_data['features']
            scaler_X = model_data['scaler_X']
            scaler_y = model_data['scaler_y']
            
            # Prepare features for prediction
            X_pred = []
            for feature in features:
                if feature in time_features:
                    X_pred.append(time_features[feature])
                elif feature in current_data:
                    X_pred.append(current_data[feature])
                else:
                    X_pred.append(0)  # Default value if feature missing
            
            # Make prediction based on model type
            try:
                if model_type == 'lstm' and scaler_X is not None and scaler_y is not None:
                    X_pred_array = np.array(X_pred).reshape(1, -1)
                    X_pred_scaled = scaler_X.transform(X_pred_array)
                    X_pred_lstm = X_pred_scaled.reshape((1, 1, X_pred_scaled.shape[1]))
                    pred_scaled = model.predict(X_pred_lstm, verbose=0)
                    pred_value = scaler_y.inverse_transform(pred_scaled).flatten()[0]
                else:
                    pred_value = model.predict([X_pred])[0]
                
                step_predictions[metric] = pred_value
                hourly_predictions[metric].append(pred_value)
            except Exception as e:
                st.warning(f"Hourly prediction failed for {metric}: {str(e)}")
                hourly_predictions[metric].append(current_data.get(metric, 0))
                step_predictions[metric] = current_data.get(metric, 0)
        
        # Roll forward: Update current_data with predictions for next iteration
        for metric in metrics:
            if metric not in step_predictions:
                continue
                
            # Capture old lag values before overwriting
            old_lag1 = current_data.get(f'{metric}_lag1', step_predictions[metric])
            old_lag2 = current_data.get(f'{metric}_lag2', step_predictions[metric])
            
            # Shift lag values
            current_data[f'{metric}_lag3'] = current_data.get(f'{metric}_lag2', old_lag1)
            current_data[f'{metric}_lag2'] = old_lag1
            current_data[f'{metric}_lag1'] = step_predictions[metric]
            
            # Update rolling statistics using distinct recent values
            recent_values = [
                step_predictions[metric],
                old_lag1,
                old_lag2
            ]
            current_data[f'{metric}_rolling_mean_3'] = np.mean(recent_values)
            current_data[f'{metric}_rolling_std_3'] = np.std(recent_values)
        
        # Update timestamp
        current_data['created_at'] = step_timestamp
    
    # Return all hourly predictions
    return {
        'timestamps': hourly_timestamps,
        'predictions': hourly_predictions
    }

def predict_all_models_hourly(models, df_features, hours_ahead=24):
    """Get hourly predictions from all three models"""
    predictions = {}
    for model_type in ['rf', 'xgb', 'lstm']:
        if model_type in models and models[model_type]:
            predictions[model_type] = predict_hourly(models, df_features, hours_ahead, model_type)
    return predictions

def predict_all_models(models, df_features, hours_ahead=1):
    """Get predictions from all three models"""
    predictions = {}
    for model_type in ['rf', 'xgb', 'lstm']:
        if model_type in models and models[model_type]:
            predictions[model_type] = predict_future(models, df_features, hours_ahead, model_type)
    return predictions

def convert_single_model_to_json(hourly_predictions, model_key, model_name):
    """Convert hourly predictions for a single model to JSON format"""
    json_output = {
        "prediction_metadata": {
            "model": model_name,
            "generated_at": datetime.now().isoformat(),
            "total_hours": len(hourly_predictions[model_key]['timestamps']) if hourly_predictions and model_key in hourly_predictions else 0
        },
        "predictions": []
    }
    
    if not hourly_predictions or model_key not in hourly_predictions:
        return json_output
    
    timestamps = hourly_predictions[model_key]['timestamps']
    predictions = hourly_predictions[model_key]['predictions']
    
    for i, timestamp in enumerate(timestamps):
        hourly_data = {
            "timestamp": timestamp.isoformat(),
            "hour_offset": i + 1,
            "air_quality_metrics": {
                "temperature": round(float(predictions['temperature'][i]), 2) if i < len(predictions['temperature']) else 0,
                "humidity": round(float(predictions['humidity'][i]), 2) if i < len(predictions['humidity']) else 0,
                "co2": round(float(predictions['co2'][i]), 2) if i < len(predictions['co2']) else 0,
                "co": round(float(predictions['co'][i]), 2) if i < len(predictions['co']) else 0,
                "pm25": round(float(predictions['pm25'][i]), 2) if i < len(predictions['pm25']) else 0,
                "pm10": round(float(predictions['pm10'][i]), 2) if i < len(predictions['pm10']) else 0
            }
        }
        json_output["predictions"].append(hourly_data)
    
    return json_output

def create_hourly_prediction_plot(df, hourly_predictions, metric, metric_name, model_type='rf'):
    """Create a plot showing hourly predictions"""
    
    # Create figure
    fig = go.Figure()
    
    # Add hourly predictions if available
    if hourly_predictions and model_type in hourly_predictions:
        pred_data = hourly_predictions[model_type]
        if pred_data and metric in pred_data['predictions']:
            timestamps = pred_data['timestamps']
            predictions = pred_data['predictions'][metric]
            
            if len(predictions) > 0:
                # Split predictions into segments for different colors
                total_hours = len(timestamps)
                third = total_hours // 3
                
                # Near term predictions (green)
                if third > 0:
                    fig.add_trace(go.Scatter(
                        x=timestamps[:third],
                        y=predictions[:third],
                        mode='lines+markers',
                        name='Near-term Forecast',
                        line=dict(color='#50C878', width=2, dash='dot'),
                        marker=dict(size=6, color='#50C878'),
                        hovertemplate='%{x|%b %d, %H:%M}<br>Predicted: %{y:.2f}<extra></extra>'
                    ))
                
                # Medium term predictions (yellow)
                if total_hours > third:
                    fig.add_trace(go.Scatter(
                        x=timestamps[third:2*third],
                        y=predictions[third:2*third],
                        mode='lines+markers',
                        name='Medium-term Forecast',
                        line=dict(color='#FFD700', width=2, dash='dot'),
                        marker=dict(size=6, color='#FFD700'),
                        hovertemplate='%{x|%b %d, %H:%M}<br>Predicted: %{y:.2f}<extra></extra>'
                    ))
                
                # Long term predictions (red/orange)
                if total_hours > 2*third:
                    fig.add_trace(go.Scatter(
                        x=timestamps[2*third:],
                        y=predictions[2*third:],
                        mode='lines+markers',
                        name='Long-term Forecast',
                        line=dict(color='#FF6B6B', width=2, dash='dot'),
                        marker=dict(size=6, color='#FF6B6B'),
                        hovertemplate='%{x|%b %d, %H:%M}<br>Predicted: %{y:.2f}<extra></extra>'
                    ))
    
    # Update layout
    fig.update_layout(
        title=f'Prediction Analysis for {metric_name}',
        xaxis_title='',
        yaxis_title=f'{metric_name}',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            tickformat='%b %d, %Y<br>%H:%M'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_all_models_prediction_plot(df, hourly_predictions, metric, metric_name):
    """Create a plot showing predictions from all three models"""
    
    # Create figure
    fig = go.Figure()
    
    # Model colors
    model_colors = {
        'rf': '#50C878',
        'xgb': '#FFD700', 
        'lstm': '#FF6B6B'
    }
    
    model_names = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'lstm': 'LSTM'
    }
    
    # Add predictions from each model
    for model_type in ['rf', 'xgb', 'lstm']:
        if hourly_predictions and model_type in hourly_predictions:
            pred_data = hourly_predictions[model_type]
            if pred_data and metric in pred_data['predictions']:
                timestamps = pred_data['timestamps']
                predictions = pred_data['predictions'][metric]
                
                if len(predictions) > 0:
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=predictions,
                        mode='lines+markers',
                        name=model_names[model_type],
                        line=dict(color=model_colors[model_type], width=2, dash='dot'),
                        marker=dict(size=5, color=model_colors[model_type]),
                        hovertemplate='%{x|%b %d, %H:%M}<br>' + model_names[model_type] + ': %{y:.2f}<extra></extra>'
                    ))
    
    # Update layout
    fig.update_layout(
        title=f'Prediction Analysis for {metric_name} - All Models Comparison',
        xaxis_title='',
        yaxis_title=f'{metric_name}',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            tickformat='%b %d, %Y<br>%H:%M'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main app
def main():
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Fetch data
    with st.spinner("Fetching data from Supabase..."):
        df = fetch_data()
    
    if df is None or len(df) == 0:
        st.error("Unable to load data from database. Please check your connection and try again.")
        st.info("""
        **Troubleshooting tips:**
        - Check if your Supabase database has data in the 'airquality' table
        - Verify the table has columns: created_at, temperature, humidity, co2, co, pm25, pm10
        - Ensure your internet connection is stable
        """)
        return
    
    st.sidebar.success(f"✅ Loaded {len(df)} records")
    st.sidebar.markdown(f"**Date Range:** {df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}")
    
    # Check required columns
    required_columns = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns in data: {missing_columns}. Some features may not work properly.")
    
    # Prediction settings
    st.sidebar.subheader("Prediction Settings")
    forecast_hours = st.sidebar.slider(
        "Forecast Hours Ahead",
        min_value=1,
        max_value=12,
        value=6,
        help="Select how many hours into the future to predict"
    )
    
    # Train models
    with st.spinner("Training machine learning models..."):
        result = train_models(df)
        
        if result is None:
            st.error("Unable to train models. Not enough data available.")
            st.info("""
            **Requirements for training:**
            - At least 10 data records
            - Data should include time-series measurements
            - All required metrics should be present
            """)
            return
        
        models, performance, df_features = result
    
    st.sidebar.success("✅ Models trained successfully")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔮 Predictions", "⏰ Hourly Predictions", "📈 Historical Trends", "📉 Model Performance"])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Recent Data")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        latest = df.iloc[-1]
        
        with col1:
            st.metric("Temperature", f"{latest['temperature']:.1f}°C")
        with col2:
            st.metric("Humidity", f"{latest['humidity']:.1f}%")
        with col3:
            st.metric("CO2", f"{latest['co2']:.0f} ppm")
        with col4:
            st.metric("CO", f"{latest['co']:.2f} ppm")
        with col5:
            st.metric("PM2.5", f"{latest['pm25']:.1f} µg/m³")
        with col6:
            st.metric("PM10", f"{latest['pm10']:.1f} µg/m³")
        
        st.subheader("Data Statistics")
        
        # Calculate statistics for air quality metrics
        metrics_cols = [col for col in required_columns if col in df.columns]
        if metrics_cols:
            stats_df = df[metrics_cols].describe().T
            stats_df = stats_df.round(2)
            stats_df.index = [f'{col} ({unit})' for col, unit in zip(
                ['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10'],
                ['°C', '%', 'ppm', 'ppm', 'µg/m³', 'µg/m³']
            ) if col.lower() in metrics_cols]
            
            st.dataframe(stats_df, use_container_width=True)
        
        st.subheader("Latest 50 Records")
        display_cols = ['created_at'] + [col for col in required_columns if col in df.columns]
        display_df = df[display_cols].tail(50)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Tab 2: Predictions
    with tab2:
        st.header("🔮 Future Predictions - Model Comparison")
        
        all_predictions = predict_all_models(models, df_features, hours_ahead=forecast_hours)
        
        if all_predictions:
            timestamp = all_predictions['rf']['timestamp']
            st.markdown(f"### Predicted values for **{timestamp.strftime('%Y-%m-%d %H:%M')}** ({forecast_hours} hours ahead)")
            
            # Model comparison metrics
            st.subheader("Predictions by Model")
            
            metrics_names = ['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10']
            metrics_keys = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
            
            # Create comparison dataframe
            comparison_data = []
            for i, (name, key) in enumerate(zip(metrics_names, metrics_keys)):
                if key in latest:
                    row = {
                        'Metric': name,
                        'Current': f"{latest[key]:.2f}",
                        'Random Forest': f"{all_predictions['rf'].get(key, 0):.2f}",
                        'XGBoost': f"{all_predictions['xgb'].get(key, 0):.2f}",
                        'LSTM': f"{all_predictions['lstm'].get(key, 0):.2f}"
                    }
                    comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Tab 3: Hourly Predictions
    with tab3:
        st.header("⏰ Hourly Predictions Analysis")
        
        st.markdown("""
        This section shows hour-by-hour predictions for all air quality parameters.
        """)
        
        # Settings for hourly predictions
        col1, col2 = st.columns(2)
        with col1:
            hourly_forecast_hours = st.selectbox(
                "Select Forecast Duration",
                options=[6, 12, 18, 24],
                index=1,
                help="Number of hours to predict into the future"
            )
        
        with col2:
            selected_model = st.selectbox(
                "Select Model",
                options=[
                    ('All Models', 'all'),
                    ('Random Forest', 'rf'),
                    ('XGBoost', 'xgb'),
                    ('LSTM', 'lstm')
                ],
                format_func=lambda x: x[0],
                help="Choose which model to use for predictions"
            )
        
        model_name, model_key = selected_model
        
        # Generate hourly predictions
        with st.spinner(f"Generating {hourly_forecast_hours}-hour predictions using {model_name}..."):
            hourly_predictions = predict_all_models_hourly(models, df_features, hours_ahead=hourly_forecast_hours)
        
        if hourly_predictions:
            st.success(f"✅ Generated predictions for the next {hourly_forecast_hours} hours")
            
            # Define all metrics
            all_metrics = [
                ('PM10', 'pm10'),
                ('PM2.5', 'pm25'),
                ('CO2', 'co2'),
                ('CO', 'co'),
                ('Temperature', 'temperature'),
                ('Humidity', 'humidity')
            ]
            
            # Create plots for each metric
            if model_key == 'all':
                # Show all models on same graph
                for metric_name, metric_key_val in all_metrics:
                    fig = create_all_models_prediction_plot(
                        df, 
                        hourly_predictions, 
                        metric_key_val, 
                        metric_name
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Show single model predictions
                for metric_name, metric_key_val in all_metrics:
                    fig = create_hourly_prediction_plot(
                        df, 
                        hourly_predictions, 
                        metric_key_val, 
                        metric_name, 
                        model_type=model_key
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Historical Trends
    with tab4:
        st.header("📈 Historical Data Visualization")
        
        available_metrics = [('Temperature', 'temperature'), ('Humidity', 'humidity'), 
                           ('CO2', 'co2'), ('CO', 'co'), ('PM2.5', 'pm25'), ('PM10', 'pm10')]
        available_metrics = [(name, key) for name, key in available_metrics if key in df.columns]
        
        if available_metrics:
            metric_select = st.selectbox(
                "Select Metric to Visualize",
                options=available_metrics,
                format_func=lambda x: x[0]
            )
            
            metric_name, metric_key = metric_select
            
            # Time series plot
            fig = px.line(
                df,
                x='created_at',
                y=metric_key,
                title=f'{metric_name} Over Time',
                labels={'created_at': 'Time', metric_key: metric_name}
            )
            
            fig.update_traces(line_color='#1f77b4', line_width=2)
            fig.update_layout(height=400, hovermode='x unified')
            
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
    
    # Tab 5: Model Performance
    with tab5:
        st.header("📉 Model Performance Comparison")
        
        st.markdown("""
        **Three Models are compared:**
        - **Random Forest**: Ensemble method using decision trees
        - **XGBoost**: Gradient boosting algorithm optimized for speed and performance
        - **LSTM**: Deep learning recurrent neural network for time-series forecasting
        
        **Evaluation Metrics:**
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values (lower is better)
        - **RMSE (Root Mean Squared Error)**: Square root of average squared differences (lower is better, penalizes large errors more)
        - **R² Score (Coefficient of Determination)**: Proportion of variance in the dependent variable predictable from independent variables (higher is better, 1.0 is perfect)
        """)
        
        # Create comprehensive performance comparison
        metrics_list = [metric for metric in ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10'] 
                       if metric in performance['rf']]
        metrics_display = [name for name, key in zip(['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10'], 
                                                    ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']) 
                          if key in metrics_list]
        
        if metrics_list:
            # MAE Comparison Table
            st.subheader("MAE Comparison Across Models")
            mae_data = []
            for i, metric in enumerate(metrics_list):
                mae_data.append({
                    'Metric': metrics_display[i],
                    'Random Forest': f"{performance['rf'][metric]['MAE']:.3f}",
                    'XGBoost': f"{performance['xgb'][metric]['MAE']:.3f}",
                    'LSTM': f"{performance['lstm'].get(metric, {}).get('MAE', 'N/A')}"
                })
            mae_df = pd.DataFrame(mae_data)
            st.dataframe(mae_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
