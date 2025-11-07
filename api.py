from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Air Quality Prediction API",
    description="API for air quality predictions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Configuration
SUPABASE_URL = "https://fjfmgndbiespptmsnrff.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"

def fetch_data():
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

@app.get("/")
async def root():
    return {
        "message": "Air Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "data": "/data",
            "predict": "/predict/{hours}"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/data")
async def get_data():
    df = fetch_data()
    if df is None:
        raise HTTPException(status_code=500, detail="No data available")
    
    return {
        "records": len(df),
        "columns": list(df.columns),
        "latest_timestamp": df['created_at'].max().isoformat() if 'created_at' in df.columns else None
    }

@app.get("/predict/{hours}")
async def predict(hours: int = 6):
    if hours < 1 or hours > 24:
        raise HTTPException(status_code=400, detail="Hours must be between 1 and 24")
    
    try:
        df = fetch_data()
        if df is None or len(df) < 10:
            raise HTTPException(status_code=400, detail="Not enough data for prediction")
        
        return {
            "prediction": f"Prediction for {hours} hours ahead",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
