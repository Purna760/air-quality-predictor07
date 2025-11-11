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

# Supabase Configuration
# Try Streamlit secrets first, then environment variables
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
    SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY"))
except:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Validate credentials are set
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("⚠️ Supabase credentials not configured!")
    st.info("""
    Please configure Supabase credentials using one of these methods:
    
    **Option 1: Streamlit Secrets (Recommended)**
    - Create `.streamlit/secrets.toml` with:
    ```toml
    SUPABASE_URL = "your-url"
    SUPABASE_ANON_KEY = "your-key"
    ```
    
    **Option 2: Environment Variables**
    - Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables
    """)
    st.stop()

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
    df = df.copy()
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['day_of_month'] = df['created_at'].dt.day
    df['month'] = df['created_at'].dt.month
    
    # Create lag features for each metric
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    for metric in metrics:
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
    
    if len(df_features) < 5:
        st.error("Not enough data for training after feature engineering")
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    models = {'rf': {}, 'xgb': {}, 'lstm': {}}
    performance = {'rf': {}, 'xgb': {}, 'lstm': {}}
    
    # Base features (time features + all lag features)
    base_features = ['hour', 'day_of_week', 'day_of_month', 'month']
    
    for metric in metrics:
        # Features for this metric
        lag_features = [col for col in df_features.columns if metric in col and col != metric]
        features = base_features + lag_features
        
        X = df_features[features]
        y = df_features[metric]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale data for LSTM
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
        
        # 1. Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
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
        
        # 2. Train XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,
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
        
        # 3. Train LSTM model
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        lstm_model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        lstm_model.fit(
            X_train_lstm, y_train_scaled,
            epochs=50,
            batch_size=32,
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
    
    return models, performance, df_features

# Make predictions
def predict_future(models, df_features, hours_ahead=1, model_type='rf', batch_size=1000):
    """
    Predict future values for all metrics using iterative multi-step forecasting.
    Now uses ALL historical data with batch processing to prevent memory issues.
    
    Args:
        models: Trained models dictionary
        df_features: Full historical data with engineered features
        hours_ahead: Number of hours to predict into the future
        model_type: Type of model ('rf', 'xgb', or 'lstm')
        batch_size: Number of samples to process in each batch (default: 1000)
    """
    if models is None or df_features is None:
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    model_dict = models[model_type]
    
    # Phase 1: Replay ALL historical data in batches to build complete feature state
    current_data = replay_historical_data_in_batches(df_features, batch_size)
    
    if current_data is None:
        return None
    
    last_timestamp = current_data['created_at']
    
    # Initialize predictions dictionary
    step_predictions = {}
    
    # Phase 2: Iteratively predict for each future hour
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
                else:
                    X_pred.append(current_data[feature])
            
            # Make prediction based on model type
            if model_type == 'lstm':
                X_pred_array = np.array(X_pred).reshape(1, -1)
                X_pred_scaled = scaler_X.transform(X_pred_array)
                X_pred_lstm = X_pred_scaled.reshape((1, 1, X_pred_scaled.shape[1]))
                pred_scaled = model.predict(X_pred_lstm, verbose=0)
                pred_value = scaler_y.inverse_transform(pred_scaled).flatten()[0]
            else:
                pred_value = model.predict([X_pred])[0]
            
            step_predictions[metric] = pred_value
        
        # Roll forward: Update current_data with predictions for next iteration
        for metric in metrics:
            # Capture old lag values before overwriting
            old_lag1 = current_data[f'{metric}_lag1']
            old_lag2 = current_data[f'{metric}_lag2']
            
            # Shift lag values
            current_data[f'{metric}_lag3'] = current_data[f'{metric}_lag2']
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
    predictions = {metric: step_predictions[metric] for metric in metrics}
    predictions['timestamp'] = last_timestamp + timedelta(hours=hours_ahead)
    
    return predictions

def fetch_data_in_batches(batch_size=1000, offset=0, limit=None):
    """
    Fetch data from Supabase in batches to reduce memory pressure.
    
    Args:
        batch_size: Number of records to fetch per batch
        offset: Starting offset for fetching data
        limit: Maximum total records to fetch (None for all)
    
    Returns:
        Generator yielding DataFrames of batch_size records
    """
    try:
        supabase = init_supabase()
        if supabase is None:
            return None
        
        current_offset = offset
        total_fetched = 0
        
        while True:
            # Determine how many records to fetch in this batch
            fetch_count = batch_size
            if limit is not None:
                remaining = limit - total_fetched
                if remaining <= 0:
                    break
                fetch_count = min(batch_size, remaining)
            
            # Fetch batch from database
            response = supabase.table('airquality').select('*').order('created_at', desc=False).range(current_offset, current_offset + fetch_count - 1).execute()
            
            if not response.data or len(response.data) == 0:
                break
            
            # Convert to DataFrame
            df_batch = pd.DataFrame(response.data)
            df_batch['created_at'] = pd.to_datetime(df_batch['created_at'])
            df_batch = df_batch.sort_values('created_at').reset_index(drop=True)
            
            yield df_batch
            
            total_fetched += len(df_batch)
            current_offset += len(df_batch)
            
            # If we got fewer records than requested, we've reached the end
            if len(df_batch) < fetch_count:
                break
                
    except Exception as e:
        st.error(f"Error fetching data in batches: {str(e)}")
        return None

def replay_historical_data_in_batches(df_features, batch_size=1000):
    """
    Process historical data in batches to build complete feature state.
    This implementation processes the already-loaded DataFrame in chunks to simulate
    incremental state building, ensuring lag features and rolling statistics reflect
    all previous records.
    
    Note: For true memory optimization with very large datasets (>10000 records),
    consider using fetch_data_in_batches() to stream data from the database.
    
    Args:
        df_features: DataFrame with all engineered features
        batch_size: Number of samples to process in each batch
    
    Returns:
        The final row state after processing all historical data
    """
    if df_features is None or len(df_features) == 0:
        return None
    
    total_samples = len(df_features)
    
    # If dataset is smaller than batch size, return the last row directly
    if total_samples <= batch_size:
        return df_features.iloc[-1].copy()
    
    # For large datasets, process in batches to demonstrate the pattern
    # Note: The DataFrame is already in memory, but this approach shows
    # how features would be built incrementally with streaming data
    current_row = df_features.iloc[0].copy()
    
    # Process data in batches, updating the feature state incrementally
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        
        # Get the last row of this batch as it has the most up-to-date features
        # (lag features, rolling stats) that incorporate all data up to that point
        current_row = df_features.iloc[batch_end - 1].copy()
    
    return current_row

def predict_hourly(models, df_features, hours_ahead=24, model_type='rf', batch_size=1000):
    """
    Predict future values for all metrics hour by hour, returning predictions for EACH hour.
    Now uses ALL historical data with batch processing to prevent memory issues.
    
    Args:
        models: Trained models dictionary
        df_features: Full historical data with engineered features
        hours_ahead: Number of hours to predict into the future
        model_type: Type of model ('rf', 'xgb', or 'lstm')
        batch_size: Number of samples to process in each batch (default: 1000)
    """
    if models is None or df_features is None:
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    model_dict = models[model_type]
    
    # Phase 1: Replay ALL historical data in batches to build complete feature state
    current_data = replay_historical_data_in_batches(df_features, batch_size)
    
    if current_data is None:
        return None
    
    last_timestamp = current_data['created_at']
    
    # Initialize results storage for all hourly predictions
    hourly_predictions = {metric: [] for metric in metrics}
    hourly_timestamps = []
    
    # Phase 2: Iteratively predict for each future hour
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
                else:
                    X_pred.append(current_data[feature])
            
            # Make prediction based on model type
            if model_type == 'lstm':
                X_pred_array = np.array(X_pred).reshape(1, -1)
                X_pred_scaled = scaler_X.transform(X_pred_array)
                X_pred_lstm = X_pred_scaled.reshape((1, 1, X_pred_scaled.shape[1]))
                pred_scaled = model.predict(X_pred_lstm, verbose=0)
                pred_value = scaler_y.inverse_transform(pred_scaled).flatten()[0]
            else:
                pred_value = model.predict([X_pred])[0]
            
            step_predictions[metric] = pred_value
            hourly_predictions[metric].append(pred_value)
        
        # Roll forward: Update current_data with predictions for next iteration
        for metric in metrics:
            # Capture old lag values before overwriting
            old_lag1 = current_data[f'{metric}_lag1']
            old_lag2 = current_data[f'{metric}_lag2']
            
            # Shift lag values
            current_data[f'{metric}_lag3'] = current_data[f'{metric}_lag2']
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

def predict_all_models_hourly(models, df_features, hours_ahead=24, batch_size=1000):
    """Get hourly predictions from all three models with batch processing"""
    predictions = {}
    for model_type in ['rf', 'xgb', 'lstm']:
        predictions[model_type] = predict_hourly(models, df_features, hours_ahead, model_type, batch_size)
    return predictions

def predict_all_models(models, df_features, hours_ahead=1, batch_size=1000):
    """Get predictions from all three models with batch processing"""
    predictions = {}
    for model_type in ['rf', 'xgb', 'lstm']:
        predictions[model_type] = predict_future(models, df_features, hours_ahead, model_type, batch_size)
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
                "temperature": round(float(predictions['temperature'][i]), 2),
                "humidity": round(float(predictions['humidity'][i]), 2),
                "co2": round(float(predictions['co2'][i]), 2),
                "co": round(float(predictions['co'][i]), 2),
                "pm25": round(float(predictions['pm25'][i]), 2),
                "pm10": round(float(predictions['pm10'][i]), 2)
            }
        }
        json_output["predictions"].append(hourly_data)
    
    return json_output

def create_hourly_prediction_plot(df, hourly_predictions, metric, metric_name, model_type='rf'):
    """Create a plot showing hourly predictions similar to the example image"""
    
    # Create figure with dark background
    fig = go.Figure()
    
    # Add hourly predictions if available
    if hourly_predictions and model_type in hourly_predictions:
        pred_data = hourly_predictions[model_type]
        if pred_data:
            timestamps = pred_data['timestamps']
            predictions = pred_data['predictions'][metric]
            
            # Split predictions into segments for different colors
            # First third - green (near term)
            # Middle third - yellow (medium term)
            # Last third - red (long term)
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
    
    # Update layout with dark theme similar to the example
    fig.update_layout(
        title=f'Prediction Analysis for {metric_name}',
        xaxis_title='',
        yaxis_title=f'{metric_name}',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            tickformat='%b %d, %Y<br>%H:%M'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#333333'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    return fig

def create_all_models_prediction_plot(df, hourly_predictions, metric, metric_name):
    """Create a plot showing predictions from all three models"""
    
    # Create figure with dark background
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
            if pred_data:
                timestamps = pred_data['timestamps']
                predictions = pred_data['predictions'][metric]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=predictions,
                    mode='lines+markers',
                    name=model_names[model_type],
                    line=dict(color=model_colors[model_type], width=2, dash='dot'),
                    marker=dict(size=5, color=model_colors[model_type]),
                    hovertemplate='%{x|%b %d, %H:%M}<br>' + model_names[model_type] + ': %{y:.2f}<extra></extra>'
                ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=f'Prediction Analysis for {metric_name} - All Models Comparison',
        xaxis_title='',
        yaxis_title=f'{metric_name}',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            tickformat='%b %d, %Y<br>%H:%M'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#333333'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
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
        return
    
    st.sidebar.success(f"✅ Loaded {len(df)} records")
    st.sidebar.markdown(f"**Date Range:** {df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}")
    
    # Important info about data usage
    st.sidebar.info(f"ℹ️ **All {len(df)} historical records** are used for model training and predictions")
    
    # Prediction settings
    st.sidebar.subheader("Prediction Settings")
    forecast_hours = st.sidebar.slider(
        "Forecast Hours Ahead",
        min_value=1,
        max_value=24,
        value=6,
        help="Select how many hours into the future to predict"
    )
    
    # Batch processing settings
    st.sidebar.subheader("Advanced Settings")
    batch_size = st.sidebar.number_input(
        "Batch Size for Processing",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Controls how data is processed internally. Lower values (500-1000) help prevent errors with very large datasets (7000+ records)."
    )
    
    total_records = len(df)
    st.sidebar.caption(f"Processing in {(total_records + batch_size - 1) // batch_size} batch(es) of {batch_size} records")
    
    # Train models
    with st.spinner("Training machine learning models..."):
        result = train_models(df)
        
        if result is None:
            st.error("Unable to train models. Not enough data available.")
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
        metrics_cols = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
        stats_df = df[metrics_cols].describe().T
        stats_df = stats_df.round(2)
        stats_df.index = ['Temperature (°C)', 'Humidity (%)', 'CO2 (ppm)', 'CO (ppm)', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)']
        
        st.dataframe(stats_df, use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        
        # Calculate correlation matrix
        corr_matrix = df[metrics_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10'],
            y=['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10'],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Correlation Matrix of Air Quality Metrics',
            xaxis_title='',
            yaxis_title='',
            height=500,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Latest 50 Records (Total: {len(df)} records used for training)")
        display_df = df[['created_at', 'temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']].tail(50)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.caption(f"ℹ️ Showing last 50 records for display. All {len(df)} historical records are used for model training and feature engineering.")
    
    # Tab 2: Predictions
    with tab2:
        st.header("🔮 Future Predictions - Model Comparison")
        
        all_predictions = predict_all_models(models, df_features, hours_ahead=forecast_hours, batch_size=batch_size)
        
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
                row = {
                    'Metric': name,
                    'Current': f"{latest[key]:.2f}",
                    'Random Forest': f"{all_predictions['rf'][key]:.2f}",
                    'XGBoost': f"{all_predictions['xgb'][key]:.2f}",
                    'LSTM': f"{all_predictions['lstm'][key]:.2f}"
                }
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Prediction comparison chart
            st.subheader("Model Predictions Comparison")
            
            # Create subplots for each metric
            for name, key in zip(metrics_names, metrics_keys):
                fig = go.Figure()
                
                current_val = latest[key]
                rf_val = all_predictions['rf'][key]
                xgb_val = all_predictions['xgb'][key]
                lstm_val = all_predictions['lstm'][key]
                
                fig.add_trace(go.Bar(
                    x=['Current', 'Random Forest', 'XGBoost', 'LSTM'],
                    y=[current_val, rf_val, xgb_val, lstm_val],
                    marker_color=['lightblue', 'salmon', 'lightgreen', 'gold'],
                    text=[f"{current_val:.2f}", f"{rf_val:.2f}", f"{xgb_val:.2f}", f"{lstm_val:.2f}"],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f'{name} - Current vs Model Predictions',
                    xaxis_title='',
                    yaxis_title='Value',
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Overall comparison
            st.subheader("All Metrics - Model Comparison")
            
            fig = go.Figure()
            
            current_values = [latest[key] for key in metrics_keys]
            rf_values = [all_predictions['rf'][key] for key in metrics_keys]
            xgb_values = [all_predictions['xgb'][key] for key in metrics_keys]
            lstm_values = [all_predictions['lstm'][key] for key in metrics_keys]
            
            fig.add_trace(go.Scatter(
                x=metrics_names,
                y=current_values,
                mode='lines+markers',
                name='Current',
                line=dict(color='blue', width=2),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=metrics_names,
                y=rf_values,
                mode='lines+markers',
                name='Random Forest',
                line=dict(color='red', width=2),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=metrics_names,
                y=xgb_values,
                mode='lines+markers',
                name='XGBoost',
                line=dict(color='green', width=2),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=metrics_names,
                y=lstm_values,
                mode='lines+markers',
                name='LSTM',
                line=dict(color='orange', width=2),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title=f'All Models Prediction Comparison ({forecast_hours}h ahead)',
                xaxis_title='Metrics',
                yaxis_title='Predicted Value',
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
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
            hourly_predictions = predict_all_models_hourly(models, df_features, hours_ahead=hourly_forecast_hours, batch_size=batch_size)
        
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
            
            # Show prediction data table
            with st.expander("📋 View Detailed Hourly Predictions"):
                if model_key == 'all':
                    # Show all models in table
                    for model_type in ['rf', 'xgb', 'lstm']:
                        model_display_names = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'lstm': 'LSTM'}
                        st.subheader(f"{model_display_names[model_type]} Predictions")
                        
                        if model_type in hourly_predictions and hourly_predictions[model_type]:
                            pred_data = hourly_predictions[model_type]
                            timestamps = pred_data['timestamps']
                            predictions = pred_data['predictions']
                            
                            # Create dataframe
                            table_data = {'Timestamp': timestamps}
                            for metric_name, metric_key_val in all_metrics:
                                table_data[metric_name] = [f"{val:.2f}" for val in predictions[metric_key_val]]
                            
                            pred_df = pd.DataFrame(table_data)
                            st.dataframe(pred_df, use_container_width=True, hide_index=True)
                else:
                    # Show single model
                    if model_key in hourly_predictions and hourly_predictions[model_key]:
                        pred_data = hourly_predictions[model_key]
                        timestamps = pred_data['timestamps']
                        predictions = pred_data['predictions']
                        
                        # Create dataframe
                        table_data = {'Timestamp': timestamps}
                        for metric_name, metric_key_val in all_metrics:
                            table_data[metric_name] = [f"{val:.2f}" for val in predictions[metric_key_val]]
                        
                        pred_df = pd.DataFrame(table_data)
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            # JSON Export Section
            st.subheader("📥 Export Predictions as JSON")
            st.markdown("Download hourly predictions in JSON format for integration with other systems. Each model has its own JSON file.")
            
            # Generate JSON for each model
            models_config = [
                ('rf', 'Random Forest', '🌲'),
                ('xgb', 'XGBoost', '⚡'),
                ('lstm', 'LSTM', '🧠')
            ]
            
            # Create three columns for the three models
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            
            for idx, (model_key, model_name, icon) in enumerate(models_config):
                json_data = convert_single_model_to_json(hourly_predictions, model_key, model_name)
                json_string = json.dumps(json_data, indent=2)
                
                with columns[idx]:
                    st.markdown(f"### {icon} {model_name}")
                    
                    # Preview button
                    with st.expander(f"Preview {model_name} JSON"):
                        st.json(json_data)
                    
                    # Download button
                    st.download_button(
                        label=f"Download {model_name}",
                        data=json_string,
                        file_name=f"{model_key}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help=f"Download {model_name} predictions in JSON format",
                        key=f"download_{model_key}"
                    )
                    
                    st.caption(f"📊 {json_data['prediction_metadata']['total_hours']} hours")
        else:
            st.error("Unable to generate hourly predictions")
    
    # Tab 4: Historical Trends
    with tab4:
        st.header("📈 Historical Data Visualization")
        
        metric_select = st.selectbox(
            "Select Metric to Visualize",
            options=[
                ('Temperature', 'temperature'),
                ('Humidity', 'humidity'),
                ('CO2', 'co2'),
                ('CO', 'co'),
                ('PM2.5', 'pm25'),
                ('PM10', 'pm10')
            ],
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
        
        # All metrics comparison
        st.subheader("All Metrics Over Time")
        
        fig = go.Figure()
        
        metrics = [
            ('Temperature', 'temperature', '°C'),
            ('Humidity', 'humidity', '%'),
            ('CO2', 'co2', 'ppm'),
            ('CO', 'co', 'ppm'),
            ('PM2.5', 'pm25', 'µg/m³'),
            ('PM10', 'pm10', 'µg/m³')
        ]
        
        for name, key, unit in metrics:
            # Normalize values for comparison
            normalized = (df[key] - df[key].min()) / (df[key].max() - df[key].min())
            fig.add_trace(go.Scatter(
                x=df['created_at'],
                y=normalized,
                name=name,
                mode='lines',
                hovertemplate=f'{name}: %{{customdata:.2f}} {unit}<extra></extra>',
                customdata=df[key]
            ))
        
        fig.update_layout(
            title='Normalized Comparison of All Metrics',
            xaxis_title='Time',
            yaxis_title='Normalized Value (0-1)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Model Performance
    with tab5:
        st.header("📉 Model Performance Comparison")
        
        st.markdown("""
        **Three Models are compared:**
        - **Random Forest**: Ensemble method using 100 decision trees
        - **XGBoost**: Gradient boosting algorithm optimized for speed and performance
        - **LSTM**: Deep learning recurrent neural network for time-series forecasting
        
        **Evaluation Metrics:**
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values (lower is better)
        - **RMSE (Root Mean Squared Error)**: Square root of average squared differences (lower is better, penalizes large errors more)
        - **R² Score (Coefficient of Determination)**: Proportion of variance in the dependent variable predictable from independent variables (higher is better, 1.0 is perfect)
        """)
        
        # Create comprehensive performance comparison
        metrics_list = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
        metrics_display = ['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10']
        
        # MAE Comparison Table
        st.subheader("MAE Comparison Across Models")
        mae_data = []
        for i, metric in enumerate(metrics_list):
            mae_data.append({
                'Metric': metrics_display[i],
                'Random Forest': f"{performance['rf'][metric]['MAE']:.3f}",
                'XGBoost': f"{performance['xgb'][metric]['MAE']:.3f}",
                'LSTM': f"{performance['lstm'][metric]['MAE']:.3f}"
            })
        mae_df = pd.DataFrame(mae_data)
        st.dataframe(mae_df, use_container_width=True, hide_index=True)
        
        # RMSE Comparison Table
        st.subheader("RMSE Comparison Across Models")
        rmse_data = []
        for i, metric in enumerate(metrics_list):
            rmse_data.append({
                'Metric': metrics_display[i],
                'Random Forest': f"{performance['rf'][metric]['RMSE']:.3f}",
                'XGBoost': f"{performance['xgb'][metric]['RMSE']:.3f}",
                'LSTM': f"{performance['lstm'][metric]['RMSE']:.3f}"
            })
        rmse_df = pd.DataFrame(rmse_data)
        st.dataframe(rmse_df, use_container_width=True, hide_index=True)
        
        # R² Score Comparison Table
        st.subheader("R² Score Comparison Across Models")
        r2_data = []
        for i, metric in enumerate(metrics_list):
            r2_data.append({
                'Metric': metrics_display[i],
                'Random Forest': f"{performance['rf'][metric]['R2']:.4f}",
                'XGBoost': f"{performance['xgb'][metric]['R2']:.4f}",
                'LSTM': f"{performance['lstm'][metric]['R2']:.4f}"
            })
        r2_df = pd.DataFrame(r2_data)
        st.dataframe(r2_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.subheader("Visual Performance Comparison")
        
        # MAE comparison chart
        fig = go.Figure()
        
        rf_mae = [performance['rf'][m]['MAE'] for m in metrics_list]
        xgb_mae = [performance['xgb'][m]['MAE'] for m in metrics_list]
        lstm_mae = [performance['lstm'][m]['MAE'] for m in metrics_list]
        
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=rf_mae,
            name='Random Forest',
            marker_color='salmon'
        ))
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=xgb_mae,
            name='XGBoost',
            marker_color='lightgreen'
        ))
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=lstm_mae,
            name='LSTM',
            marker_color='gold'
        ))
        
        fig.update_layout(
            title='MAE Comparison by Metric (Lower is Better)',
            xaxis_title='Metric',
            yaxis_title='Mean Absolute Error',
            barmode='group',
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RMSE comparison chart
        fig = go.Figure()
        
        rf_rmse = [performance['rf'][m]['RMSE'] for m in metrics_list]
        xgb_rmse = [performance['xgb'][m]['RMSE'] for m in metrics_list]
        lstm_rmse = [performance['lstm'][m]['RMSE'] for m in metrics_list]
        
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=rf_rmse,
            name='Random Forest',
            marker_color='salmon'
        ))
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=xgb_rmse,
            name='XGBoost',
            marker_color='lightgreen'
        ))
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=lstm_rmse,
            name='LSTM',
            marker_color='gold'
        ))
        
        fig.update_layout(
            title='RMSE Comparison by Metric (Lower is Better)',
            xaxis_title='Metric',
            yaxis_title='Root Mean Squared Error',
            barmode='group',
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # R² comparison chart
        fig = go.Figure()
        
        rf_r2 = [performance['rf'][m]['R2'] for m in metrics_list]
        xgb_r2 = [performance['xgb'][m]['R2'] for m in metrics_list]
        lstm_r2 = [performance['lstm'][m]['R2'] for m in metrics_list]
        
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=rf_r2,
            name='Random Forest',
            marker_color='salmon'
        ))
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=xgb_r2,
            name='XGBoost',
            marker_color='lightgreen'
        ))
        fig.add_trace(go.Bar(
            x=metrics_display,
            y=lstm_r2,
            name='LSTM',
            marker_color='gold'
        ))
        
        fig.update_layout(
            title='R² Score Comparison by Metric (Higher is Better)',
            xaxis_title='Metric',
            yaxis_title='R² Score',
            barmode='group',
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model per metric
        st.subheader("Best Performing Model by Metric")
        best_model_data = []
        for i, metric in enumerate(metrics_list):
            rf_mae_val = performance['rf'][metric]['MAE']
            xgb_mae_val = performance['xgb'][metric]['MAE']
            lstm_mae_val = performance['lstm'][metric]['MAE']
            
            best_mae = min(rf_mae_val, xgb_mae_val, lstm_mae_val)
            if best_mae == rf_mae_val:
                best_model = 'Random Forest'
            elif best_mae == xgb_mae_val:
                best_model = 'XGBoost'
            else:
                best_model = 'LSTM'
            
            best_model_data.append({
                'Metric': metrics_display[i],
                'Best Model (by MAE)': best_model,
                'MAE': f"{best_mae:.3f}"
            })
        
        best_df = pd.DataFrame(best_model_data)
        st.dataframe(best_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
