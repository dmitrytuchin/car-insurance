"""
Streamlit app for Insurance Premium Prediction.
Loads a trained model and allows users to input features for prediction.
Also shows model performance metrics and dataset insights.
Auto-trains the model if it doesn't exist.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import sys

# ──────────────────────── Page Config ────────────────────────
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",  # Auto collapse on mobile
)

# ──────────────────────── Custom CSS ────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        box-sizing: border-box;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: clamp(1.5rem, 5vw, 2.2rem);
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: clamp(0.9rem, 4vw, 1.05rem);
        opacity: 0.9;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-card .value {
        font-size: clamp(1.2rem, 4vw, 2rem);
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-card .label {
        font-size: clamp(0.75rem, 2.5vw, 0.9rem);
        color: #555;
        margin-top: 0.3rem;
        line-height: 1.2;
    }

    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
        margin: 1.5rem 0;
    }
    
    .prediction-box .amount {
        font-size: clamp(2rem, 6vw, 3rem);
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .prediction-box .subtitle {
        font-size: clamp(0.9rem, 3vw, 1.1rem);
        opacity: 0.9;
    }

    .sidebar .sidebar-content {
        background: #f8f9fc;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #e8ecf4 100%);
    }

    .model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: clamp(0.7rem, 2vw, 0.85rem);
        font-weight: 600;
    }

    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        font-size: clamp(0.75rem, 2vw, 0.95rem);
    }
    
    .comparison-table th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: clamp(8px, 2vw, 12px) clamp(8px, 2vw, 16px);
        font-weight: 600;
    }
    
    .comparison-table td {
        padding: clamp(6px, 1.5vw, 10px) clamp(6px, 1.5vw, 16px);
        border-bottom: 1px solid #eee;
    }
    
    /* Mobile header - slides from top */
    @media (max-width: 768px) {
        /* Hide default sidebar */
        div[data-testid="stSidebar"] {
            background: transparent !important;
            border: none !important;
        }
        
        /* Create sticky header container */
        .mobile-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(180deg, #f8f9fc 0%, #e8ecf4 100%);
            z-index: 1000;
            width: 100vw;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            animation: slideDown 0.3s ease-out;
            padding-bottom: 1rem;
        }
        
        @keyframes slideDown {
            from {
                transform: translateY(-100%);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
    }
    
    @keyframes slideUp {
        from {
            transform: translateY(0);
            opacity: 1;
        }
        to {
            transform: translateY(-100%);
            opacity: 0;
        }
    }
        .main-header {
            padding: 1.2rem 1rem;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            padding: 0.8rem;
            min-height: 80px;
        }
        
        .prediction-box {
            padding: 1.2rem 1rem;
            margin: 1rem 0;
        }
        
        /* Make dataframe and tables scrollable */
        .streamlit-expanderHeader {
            font-size: 0.95rem !important;
        }
    }

    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.3rem;
        }
        
        .main-header p {
            font-size: 0.85rem;
        }
        
        .metric-card {
            padding: 0.6rem;
            min-height: 70px;
        }
        
        .metric-card .value {
            font-size: 1.1rem;
        }
        
        .metric-card .label {
            font-size: 0.7rem;
        }
        
        .prediction-box {
            padding: 1rem;
        }
        
        .prediction-box .amount {
            font-size: 1.8rem;
        }
        
        .prediction-box .subtitle {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────── Load Model ────────────────────────
@st.cache_resource
def train_model_if_needed():
    """Train the model if it doesn't exist."""
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    
    if not os.path.exists(model_path):
        st.info("🚀 Training model for the first time... This may take a minute.")
        
        # Get the directory where this script is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(app_dir, "train_model.py")
        
        # Run the training script
        try:
            result = subprocess.run(
                [sys.executable, train_script],
                cwd=app_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                st.error(f"Training failed: {result.stderr}")
                return None
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    return model_path


@st.cache_resource
def load_model():
    model_path = train_model_if_needed()
    if model_path is None or not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "challenging_insurance_dataset_regression.csv",
    )
    return pd.read_csv(csv_path)


def create_feature_interactions(X: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Create polynomial and interaction features for numerical columns."""
    X_enhanced = X.copy()
    
    # Create polynomial features for key numerical cols
    for col in numerical_cols[:3]:  # Top 3 numerical features
        if col in X_enhanced.columns:
            X_enhanced[f"{col}_squared"] = X_enhanced[col] ** 2
            X_enhanced[f"{col}_sqrt"] = np.sqrt(np.abs(X_enhanced[col]))
    
    # Create interaction terms
    if len(numerical_cols) >= 2:
        col1, col2 = numerical_cols[0], numerical_cols[1]
        if col1 in X_enhanced.columns and col2 in X_enhanced.columns:
            X_enhanced[f"{col1}_x_{col2}"] = X_enhanced[col1] * X_enhanced[col2]
    
    return X_enhanced


try:
    artifact = load_model()
    model = artifact["model"]
    numerical_cols = artifact["numerical_cols"]
    categorical_cols = artifact["categorical_cols"]
    feature_names = artifact["feature_names"]
    best_name = artifact["best_name"]
    test_rmse = artifact["test_rmse"]
    results = artifact["results"]
    model_loaded = True
    # Handle both old and new format
    if results and "Best Params" not in results[0]:
        # New format - add dummy Best Params
        for r in results:
            r["Best Params"] = {}
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ──────────────────────── Header ────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Insurance Premium Predictor</h1>
    <p>Advanced ML-powered prediction engine for vehicle insurance premiums</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        f"⚠️ Model not found. Please run `python train_model.py` first.\n\n"
        f"Error: {model_error}"
    )
    st.stop()

# ──────────────────────── Detect Mobile & Load Dataset ────────────────────────
df = load_dataset()

# Check if we're on mobile via CSS media query workaround
# Initialize session state for mobile drawer
if "show_mobile_drawer" not in st.session_state:
    st.session_state.show_mobile_drawer = False

# ──────────────────────── Mobile Top Drawer or Desktop Sidebar ────────────────────────
# Create a container for mobile header/drawer
mobile_drawer_container = st.container()

# Desktop Sidebar (traditional)
with st.sidebar:
    # Desktop sidebar content
    st.markdown("## 🎛️ Input Parameters")
    st.markdown("Adjust vehicle details below to get a premium prediction.")
    st.markdown("---")

    vehicle_age = st.slider(
        "🚗 Vehicle Age (years)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
        key="desktop_vehicle_age"
    )

    vehicle_types = sorted(df["vehicle_type"].dropna().unique().tolist())
    vehicle_type = st.selectbox("🏎️ Vehicle Type", vehicle_types, index=0, key="desktop_vehicle_type")

    km_run = st.number_input(
        "📏 Kilometers Run",
        min_value=0,
        max_value=200000,
        value=50000,
        step=1000,
        key="desktop_km_run"
    )

    num_claims = st.slider(
        "📋 Number of Claims",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
        key="desktop_num_claims"
    )

    regions = sorted(df["region"].dropna().unique().tolist())
    region = st.selectbox("🌍 Region", regions, index=0, key="desktop_region")

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Premium", use_container_width=True, type="primary", key="desktop_predict")

# Mobile drawer - overlay from top
with mobile_drawer_container:
    st.markdown("""
    <style>
        .mobile-drawer-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 24px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        @media (max-width: 768px) {
            .mobile-drawer-btn {
                display: flex !important;
            }
        }
        
        @media (min-width: 769px) {
            .mobile-drawer-btn {
                display: none !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Mobile drawer toggle button
    col1, col2, col3 = st.columns([1, 1, 0.8])
    with col3:
        if st.button("⚙️ Menu", key="mobile_menu_btn", use_container_width=True):
            st.session_state.show_mobile_drawer = not st.session_state.show_mobile_drawer
    
    # Show mobile drawer if toggled
    if st.session_state.show_mobile_drawer:
        st.markdown("""
        <div style="background: linear-gradient(180deg, #f8f9fc 0%, #e8ecf4 100%); padding: 1.5rem; border-radius: 0 0 12px 12px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        
        col_close = st.columns([1, 6])
        with col_close[1]:
            if st.button("✕ Close", use_container_width=True, key="close_drawer"):
                st.session_state.show_mobile_drawer = False
        
        st.markdown("---")
        
        # Mobile input controls
        st.markdown("#### 🚗 Vehicle Age")
        vehicle_age_m = st.slider(
            "Years",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.1,
            key="mobile_vehicle_age"
        )

        st.markdown("#### 🏎️ Vehicle Type")
        vehicle_type_m = st.selectbox("Type", vehicle_types, index=0, key="mobile_vehicle_type")

        st.markdown("#### 📏 Kilometers Run")
        km_run_m = st.number_input(
            "KM",
            min_value=0,
            max_value=200000,
            value=50000,
            step=1000,
            key="mobile_km_run"
        )

        st.markdown("#### 📋 Number of Claims")
        num_claims_m = st.slider(
            "Claims",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            key="mobile_num_claims"
        )

        st.markdown("#### 🌍 Region")
        region_m = st.selectbox("Region", regions, index=0, key="mobile_region")

        st.markdown("---")
        predict_btn_m = st.button("🔮 Predict Premium", use_container_width=True, type="primary", key="mobile_predict")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Use mobile values if drawer is open and predict was clicked, else use desktop
    if st.session_state.show_mobile_drawer and "mobile_predict" in st.session_state and st.session_state.get("mobile_predict"):
        final_vehicle_age = vehicle_age_m
        final_vehicle_type = vehicle_type_m
        final_km_run = km_run_m
        final_num_claims = num_claims_m
        final_region = region_m
        show_prediction = True
    elif predict_btn:
        final_vehicle_age = vehicle_age
        final_vehicle_type = vehicle_type
        final_km_run = km_run
        final_num_claims = num_claims
        final_region = region
        show_prediction = True
    else:
        show_prediction = False


# ──────────────────────── Main Content ────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Insights"])

# ── Tab 1: Prediction ──
with tab1:
    if show_prediction:
        # Create input data with basic features
        input_data = pd.DataFrame([{
            "vehicle_age_years": final_vehicle_age,
            "vehicle_type": final_vehicle_type,
            "no_of_kilometers_run": float(final_km_run),
            "number_of_claims": float(final_num_claims),
            "region": final_region,
        }])
        
        # Apply feature engineering
        numerical_cols_basic = ["vehicle_age_years", "no_of_kilometers_run", "number_of_claims"]
        input_data_enhanced = create_feature_interactions(input_data, numerical_cols_basic)
        
        try:
            prediction = model.predict(input_data_enhanced)[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Try refreshing the page or check the logs for details.")
            st.stop()

        st.markdown(f"""
        <div class="prediction-box">
            <div class="subtitle">Estimated Insurance Premium</div>
            <div class="amount">${prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📝 Input Summary")
        
        # Mobile-responsive layout
        col1, col2 = st.columns(2)
        with col1:
            col1a, col1b = st.columns(2)
            with col1a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{final_vehicle_age:.1f}</div>
                    <div class="label">Vehicle Age (yrs)</div>
                </div>""", unsafe_allow_html=True)
            with col1b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{final_vehicle_type}</div>
                    <div class="label">Vehicle Type</div>
                </div>""", unsafe_allow_html=True)
        
        with col2:
            col2a, col2b = st.columns(2)
            with col2a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{final_num_claims}</div>
                    <div class="label">Claims</div>
                </div>""", unsafe_allow_html=True)
            with col2b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{final_region}</div>
                    <div class="label">Region</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{final_km_run:,}</div>
            <div class="label">Kilometers Run</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Adjust parameters in the sidebar (desktop) or tap ⚙️ Menu (mobile) to make a prediction.")

# ── Tab 2: Model Performance ──
with tab2:
    st.markdown(f'### Best Model: <span class="model-badge">{best_name}</span>', unsafe_allow_html=True)

    # Responsive metric cards for mobile
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{test_rmse:.2f}</div>
            <div class="label">Test RMSE</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        # find R² from results
        best_r2 = [r for r in results if r["Model"] == best_name][0]["Test R2"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{best_r2:.4f}</div>
            <div class="label">Test R²</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 🏆 Model Comparison")
    comp_df = pd.DataFrame(results)
    # Remove Best Params column if it exists and is empty for display
    if "Best Params" in comp_df.columns:
        comp_df = comp_df.drop(columns=["Best Params"])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("### 🔧 Model Details")
    best_result = [r for r in results if r["Model"] == best_name][0]
    if best_result.get("Best Params") and best_result["Best Params"]:
        params_df = pd.DataFrame(
            [{"Parameter": k.replace("model__", ""), "Value": v} for k, v in best_result["Best Params"].items()]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    else:
        st.info("✓ Pre-optimized model parameters (feature engineering enabled)")

# ── Tab 3: Data Insights ──
with tab3:
    df = load_dataset()

    st.markdown("### 📋 Dataset Overview")
    # Mobile-responsive columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(df):,}</div>
            <div class="label">Total Records</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{df.shape[1]}</div>
            <div class="label">Features</div>
        </div>""", unsafe_allow_html=True)
    
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{missing_pct:.1f}%</div>
        <div class="label">Missing Values</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 📊 Target Distribution")
    st.bar_chart(df["insurance_premium"].dropna().values)

    st.markdown("### 🔍 Premium by Vehicle Type")
    vt_stats = df.groupby("vehicle_type")["insurance_premium"].agg(["mean", "median", "std", "count"]).round(2)
    vt_stats.columns = ["Mean", "Median", "Std Dev", "Count"]
    st.dataframe(vt_stats, use_container_width=True)

    st.markdown("### 🌍 Premium by Region")
    rg_stats = df.groupby("region")["insurance_premium"].agg(["mean", "median", "std", "count"]).round(2)
    rg_stats.columns = ["Mean", "Median", "Std Dev", "Count"]
    st.dataframe(rg_stats, use_container_width=True)

    st.markdown("### 📈 Sample Data")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
