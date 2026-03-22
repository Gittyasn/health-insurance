import streamlit as st
import pandas as pd
import joblib
import os
import json
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Insurance Predictor Pro",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
    }
    
    /* Dark Metrics Styling */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(8px);
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Force visibility for dark theme text */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important; /* Brighter blue */
    }
    [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
    }
    
    .prediction-card {
        background: rgba(6, 78, 59, 0.4);
        padding: 25px;
        border-radius: 16px;
        border-left: 8px solid #10b981;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load models and info
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('model/best_insurance_model.pkl')
        encoders = joblib.load('model/encoders.pkl')
        features = joblib.load('model/features.pkl')
        with open('model/model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, encoders, features, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

model, encoders, features, model_info = load_model_artifacts()

# Sidebar Navigation
with st.sidebar:
    st.title("System Menu")
    app_mode = st.radio("Choose Section", ["🏠 Home Dashboard", "🔍 Individual Estimation", "📈 Data Trends", "📂 Data Batch Processing"])
    st.info("Analytical system for estimating annual medical premiums.")

if app_mode == "🏠 Home Dashboard" and model_info:
    st.title("Medical Insurance Estimation Dashboard")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Method", "Statistical Regression")
    col2.metric("Significance (R²)", f"{model_info['metrics']['Significance (R2)']:.4f}")
    col3.metric("Error Margin", f"${model_info['metrics']['Error Margin (MAE)']:,.2f}")

    st.subheader("Method Comparison & Accuracy")
    metrics_df = pd.DataFrame(model_info['all_metrics']).T.reset_index()
    # Use the same new key names in the chart
    fig = px.bar(metrics_df, x='index', y='Significance (R2)', color='Significance (R2)', 
                 title="Performance Comparison (R²)", labels={"index": "Method", "Significance (R2)": "R²"},
                 template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif app_mode == "🔍 Individual Estimation" and model:
    st.title("🔍 Estimate Annual Charges")
    st.markdown("Provide the relevant demographic data to generate an estimation.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Yearly Age", 18, 100, 30)
            sex = st.selectbox("Recorded Sex", encoders['sex'].classes_)
            bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0, help="Healthy BMI typically ranges from 18.5 to 24.9")
        
        with col2:
            children = st.select_slider("Dependents Count", options=list(range(10)))
            smoker = st.radio("Smoker Status", encoders['smoker'].classes_)
            region = st.selectbox("Regional Area", encoders['region'].classes_)

    if st.button("Calculate Expected Charges"):
        # Preprocess
        sex_enc = encoders['sex'].transform([sex])[0]
        smoker_enc = encoders['smoker'].transform([smoker])[0]
        region_enc = encoders['region'].transform([region])[0]
        
        input_data = pd.DataFrame([[age, sex_enc, bmi, children, smoker_enc, region_enc]], columns=features)
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Calculation Summary</h3>
                <h2 style="color: #059669;">Estimated Annual Charges: ${prediction:,.2f}</h2>
                <p>Based on our statistical analysis, the expected annual premium for a {age}-year-old {sex.lower()} 
                {'who smokes' if smoker=='yes' else 'who does not smoke'} with a BMI of {bmi:.1f} is 
                approximately <b>${prediction:,.2f}</b>.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Explain why (Simplified importance display)
        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            if smoker == 'yes':
                st.error("❗ **Significant Factor: Smoking**")
                st.write("Historic data shows a sharp increase in charges for residents recorded as smokers.")
            else:
                st.success("✅ **Positive Factor: Non-Smoker**")
                st.write("Being a non-smoker is a major contributing factor in low-premium eligibility.")
        with colB:
            if bmi > 30:
                st.warning("⚖️ **Note: High BMI**")
                st.write("BMI scores in the obese range correlate with higher healthcare charges.")
            else:
                st.info("🎯 **Note: Normal BMI Range**")
                st.write("Maintaining a standard BMI correlates with reduced risk and lower costs.")

elif app_mode == "📈 Data Trends":
    st.title("📈 Statistical Insights & Data Trends")
    st.markdown("Understand the key variables affecting medical premiums.")
    
    # Load raw data for interactive plots
    @st.cache_data
    def get_raw_data():
        if os.path.exists('insurance.csv'): return pd.read_csv('insurance.csv')
        return None
    
    raw_df = get_raw_data()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Categorical Analysis", "Interactive View", "Feature Correlation"])
    
    with tab1:
        st.subheader("Distribution of Annual Charges")
        if os.path.exists("notebook/plots/target_distribution.png"):
            st.image("notebook/plots/target_distribution.png", use_container_width=True)
        else: st.warning("Trend plot not available. Run full analysis first.")

    with tab2:
        st.subheader("Data Categorization")
        if os.path.exists("notebook/plots/pairplot.png"):
            st.image("notebook/plots/pairplot.png", use_container_width=True)
        else: st.warning("Analytical plot not available. Run full analysis first.")

    with tab3:
        if raw_df is not None:
            st.subheader("🔍 Interactive Variable Analysis")
            col_x = st.selectbox("Analyze Factor", ["age", "bmi", "children"], index=1)
            fig_int = px.scatter(raw_df, x=col_x, y='charges', color='smoker', 
                                 size='bmi', hover_data=['region'], 
                                 title=f"{col_x.capitalize()} vs Charges",
                                 template="plotly_dark")
            st.plotly_chart(fig_int, use_container_width=True)
        else: st.error("CSV source not found for interactive analysis.")

    with tab4:
        st.subheader("Variable Correlations")
        if os.path.exists("notebook/plots/correlation_heatmap.png"):
            st.image("notebook/plots/correlation_heatmap.png", use_container_width=True)

elif app_mode == "📂 Data Batch Processing":
    st.title("📂 Batch Data Processing")
    st.markdown("Process legacy data for bulk record analysis.")
    
    uploaded_file = st.file_uploader("Upload Data File (CSV)", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(raw_df)} records.")
        st.dataframe(raw_df.head(), use_container_width=True)
        
        if st.button("Calculate Batch Estimates"):
            with st.spinner("Processing records..."):
                try:
                    df_proc = raw_df.copy()
                    for col in ['sex', 'smoker', 'region']:
                        if col in df_proc.columns:
                            df_proc[col] = encoders[col].transform(df_proc[col])
                    
                    X_batch = df_proc[features]
                    raw_df['estimated_charges'] = model.predict(X_batch)
                    
                    st.divider()
                    st.subheader("✅ Processed Output")
                    st.dataframe(raw_df, use_container_width=True)
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        fig_batch = px.histogram(raw_df, x="estimated_charges", marginal="box", 
                                                 title="Distribution of Estimates", color_discrete_sequence=['#3b82f6'],
                                                 template="plotly_dark")
                        st.plotly_chart(fig_batch, use_container_width=True)
                    with col_res2:
                        csv = raw_df.to_csv(index=False).encode('utf-8')
                        st.markdown("### Export Results")
                        st.download_button("📥 Click to Download Result CSV", csv, "batch_results_processed.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error during calculation: {e}. Ensure data matches required format.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; font-size: 0.9em; padding: 20px;">
        Statistical Prediction System | © 2026 Analytical Solutions Inc.
    </div>
    """, 
    unsafe_allow_html=True
)
