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
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page configuration
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🌍 Air Quality Prediction Dashboard")
st.markdown("Predict future air quality metrics using **Random Forest, XGBoost, and LSTM** models")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    try:
        # Try to get from secrets first
        try:
            SUPABASE_URL = st.secrets["SUPABASE_URL"]
            SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
        except:
            # Fallback to environment variables
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            st.error("❌ Supabase credentials not found!")
            st.info("""
            Please add your Supabase credentials to `.streamlit/secrets.toml`:
            ```toml
            SUPABASE_URL = "your-url"
            SUPABASE_ANON_KEY = "your-key"
            ```
            """)
            return None
            
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("✅ Connected to Supabase successfully!")
        return supabase
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase: {str(e)}")
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
            df = df.sort_values('created_at').reset_index(drop=True)
            return df
        else:
            st.warning("⚠️ No data found in the database")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None

# Rest of your existing functions remain the same...
# [Include all the existing functions from your original app.py here]
# create_time_features, train_models, predict_future, predict_hourly, etc.

def main():
    # Check if secrets are loaded
    try:
        if 'SUPABASE_URL' not in st.secrets or 'SUPABASE_ANON_KEY' not in st.secrets:
            st.error("🔐 Supabase secrets not found!")
            st.info("""
            Please create a `.streamlit/secrets.toml` file with:
            ```toml
            SUPABASE_URL = "https://fjfmgndbiespptmsnrff.supabase.co"
            SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"
            ```
            """)
            return
    except:
        st.error("🔐 Streamlit secrets not accessible")
        return

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Fetch data
    with st.spinner("📡 Fetching data from Supabase..."):
        df = fetch_data()
    
    if df is None or len(df) == 0:
        st.error("❌ Unable to load data from database")
        st.info("""
        Possible issues:
        1. Supabase credentials are incorrect
        2. No data in the 'airquality' table
        3. Network connection issues
        """)
        return
    
    st.sidebar.success(f"✅ Loaded {len(df)} records")
    
    # Show basic data info
    st.sidebar.markdown(f"**Date Range:** {df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}")
    
    # Simple data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.tail(10), use_container_width=True)
    
    st.info("🚀 Application is working! The full ML prediction features will load once all dependencies are installed.")

if __name__ == "__main__":
    main()
