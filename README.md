####  Health Insurance Charges Prediction System

A professional Machine Learning project for predicting health insurance premiums based on customer demographics and lifestyle factors.

##  Features
- **Exploratory Data Analysis (EDA)**: Comprehensive univariate and bivariate analysis of health factors.
- **Multiple ML Models**: Support for Linear Regression, Random Forest, and **XGBoost**.
- **Performance Tracking**: Automated evaluation with R², MAE, and RMSE metrics.
- **Interactive Dashboard**: A premium Streamlit application for:
    - **Single Predictions**: Real-time estimation with visual indicators.
    - **Batch Processing**: CSV upload support for bulk predictions.
    - **Insights**: Interactive visualizations of data trends.
- **Explainable AI**: Visual feature importance tracking.

##  Project Structure
- `app.py`: Main Streamlit application.
- `notebook/analysis.py`: Refactored model training and evaluation script.
- `notebook/plots/`: Directory containing various analytical visualizations.
- `model/`: Saved models, encoders, and JSON metadata.
- `insurance.csv`: Primary dataset.
- `requirements.txt`: Python dependencies.

## How to Use
1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train Model**:
   ```bash
   python notebook/analysis.py
   ```
3. **Launch App**:
   ```bash
   streamlit run app.py
   ```

##  Performance
- **Selected Model**: XGBoost
- **R² Score**: ~0.89
- **Mean Absolute Error (MAE)**: ~$2,500

---
*Developed for health insurance analytics and prediction.*
