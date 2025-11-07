from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI(
    title="Air Quality Prediction API",
    description="API for hourly air quality predictions using Random Forest, XGBoost, and LSTM models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Configuration - Hardcoded with correct credentials
SUPABASE_URL = "https://fjfmgndbiespptmsnrff.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"

def fetch_data():
    """Fetch air quality data from Supabase"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = supabase.table('airquality').select('*').order('created_at', desc=False).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at').reset_index(drop=True)
            return df
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

def create_time_features(df):
    """Create time-based features for better predictions"""
    if df is None or len(df) == 0:
        return df
        
    df = df.copy()
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['day_of_month'] = df['created_at'].dt.day
    df['month'] = df['created_at'].dt.month
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    for metric in metrics:
        if metric in df.columns:
            df[f'{metric}_lag1'] = df[metric].shift(1)
            df[f'{metric}_lag2'] = df[metric].shift(2)
            df[f'{metric}_lag3'] = df[metric].shift(3)
            df[f'{metric}_rolling_mean_3'] = df[metric].rolling(window=3, min_periods=1).mean()
            df[f'{metric}_rolling_std_3'] = df[metric].rolling(window=3, min_periods=1).std()
    
    df = df.dropna()
    return df

def train_model_for_api(df, model_type='rf'):
    """Train a specific model for API"""
    if df is None or len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough data for training")
    
    df_features = create_time_features(df)
    
    if df_features is None or len(df_features) < 5:
        raise HTTPException(status_code=400, detail="Not enough data after feature engineering")
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    models = {}
    base_features = ['hour', 'day_of_week', 'day_of_month', 'month']
    
    for metric in metrics:
        if metric not in df_features.columns:
            continue
            
        lag_features = [col for col in df_features.columns if metric in col and col != metric]
        features = base_features + lag_features
        
        # Ensure all features exist
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
        
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            models[metric] = {'model': model, 'features': features, 'scaler_X': None, 'scaler_y': None}
            
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=50, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, verbose=False)
            models[metric] = {'model': model, 'features': features, 'scaler_X': None, 'scaler_y': None}
            
        elif model_type == 'lstm':
            if len(X_train) > 20:
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                
                X_train_scaled = scaler_X.fit_transform(X_train)
                y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
                
                X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                
                model = Sequential([
                    LSTM(32, activation='relu', return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                    Dropout(0.2),
                    LSTM(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                model.fit(X_train_lstm, y_train_scaled, epochs=30, batch_size=16, 
                         validation_split=0.2, callbacks=[early_stop], verbose=0)
                
                models[metric] = {'model': model, 'features': features, 'scaler_X': scaler_X, 'scaler_y': scaler_y}
    
    return models, df_features

def predict_hourly_api(models, df_features, hours_ahead=12, model_type='rf'):
    """Predict future values for all metrics hour by hour"""
    if models is None or df_features is None or len(df_features) == 0:
        raise HTTPException(status_code=500, detail="Models not available")
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    current_data = df_features.iloc[-1].copy()
    last_timestamp = current_data['created_at']
    
    hourly_predictions = {metric: [] for metric in metrics}
    hourly_timestamps = []
    
    for step in range(1, hours_ahead + 1):
        step_timestamp = last_timestamp + timedelta(hours=step)
        hourly_timestamps.append(step_timestamp)
        
        time_features = {
            'hour': step_timestamp.hour,
            'day_of_week': step_timestamp.dayofweek,
            'day_of_month': step_timestamp.day,
            'month': step_timestamp.month
        }
        
        step_predictions = {}
        for metric in metrics:
            if metric not in models:
                hourly_predictions[metric].append(0)
                continue
                
            model_data = models[metric]
            model = model_data['model']
            features = model_data['features']
            scaler_X = model_data['scaler_X']
            scaler_y = model_data['scaler_y']
            
            X_pred = []
            for feature in features:
                if feature in time_features:
                    X_pred.append(time_features[feature])
                elif feature in current_data:
                    X_pred.append(current_data[feature])
                else:
                    X_pred.append(0)
            
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
        
        for metric in metrics:
            if metric not in step_predictions:
                continue
                
            old_lag1 = current_data.get(f'{metric}_lag1', step_predictions[metric])
            old_lag2 = current_data.get(f'{metric}_lag2', step_predictions[metric])
            
            current_data[f'{metric}_lag3'] = current_data.get(f'{metric}_lag2', old_lag1)
            current_data[f'{metric}_lag2'] = old_lag1
            current_data[f'{metric}_lag1'] = step_predictions[metric]
            
            recent_values = [step_predictions[metric], old_lag1, old_lag2]
            current_data[f'{metric}_rolling_mean_3'] = np.mean(recent_values)
            current_data[f'{metric}_rolling_std_3'] = np.std(recent_values)
        
        current_data['created_at'] = step_timestamp
    
    return hourly_timestamps, hourly_predictions

def generate_json_response(model_type, model_name, hours=12):
    """Generate JSON response for a specific model"""
    df = fetch_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Unable to fetch data")
    
    models, df_features = train_model_for_api(df, model_type)
    
    if not models:
        raise HTTPException(status_code=500, detail="No models could be trained")
        
    timestamps, predictions = predict_hourly_api(models, df_features, hours, model_type)
    
    json_output = {
        "prediction_metadata": {
            "model": model_name,
            "generated_at": datetime.now().isoformat(),
            "total_hours": len(timestamps)
        },
        "predictions": []
    }
    
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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Air Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "random_forest": "/api/predictions/random-forest?hours=12",
            "xgboost": "/api/predictions/xgboost?hours=12",
            "lstm": "/api/predictions/lstm?hours=12"
        }
    }

@app.get("/api/predictions/random-forest")
async def predict_random_forest(hours: int = 12):
    """Get hourly predictions from Random Forest model"""
    try:
        result = generate_json_response('rf', 'Random Forest', hours)
        return JSONResponse(content=result)
    except HTTPException as exc:
        raise exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/xgboost")
async def predict_xgboost(hours: int = 12):
    """Get hourly predictions from XGBoost model"""
    try:
        result = generate_json_response('xgb', 'XGBoost', hours)
        return JSONResponse(content=result)
    except HTTPException as exc:
        raise exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/lstm")
async def predict_lstm(hours: int = 12):
    """Get hourly predictions from LSTM model"""
    try:
        result = generate_json_response('lstm', 'LSTM', hours)
        return JSONResponse(content=result)
    except HTTPException as exc:
        raise exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
