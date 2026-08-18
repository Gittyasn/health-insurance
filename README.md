# 🩺 Health Insurance Charges Prediction System

A professional Machine Learning project for predicting health insurance premiums based on customer demographics and lifestyle factors.

---

##  Features
- **Exploratory Data Analysis (EDA)**: Comprehensive univariate and bivariate analysis of health factors (saving 9 analytical plots).
- **Multiple ML Models**: Support for Linear Regression, Random Forest, and **XGBoost**.
- **Performance Tracking**: Automated evaluation with $R^2$, MAE, and RMSE metrics.
- **Interactive Dashboard**: A premium, dark-themed Streamlit application for:
    - **🏠 Home Dashboard**: Comparison of models and performance metrics.
    - **🔍 Individual Estimation**: Real-time estimation with AI-guided health suggestions.
    - **📈 Data Trends**: Interactive Scatter, Distribution, Categorical, and Correlation plots.
    - **📂 Batch Processing**: Upload raw CSV data to calculate bulk estimates and download as a processed CSV.
- **Explainable AI**: Real-time explanation and warning flags based on user input.

---

## 📁 Project Structure
- `app.py`: Main Streamlit application with a custom premium theme.
- `notebook/analysis.py`: Data cleaning, EDA visualization generation, model training, and evaluation script.
- `notebook/plots/`: Directory containing generated dark-themed visualizations.
- `model/`: Serialized best model (`best_insurance_model.pkl`), LabelEncoders (`encoders.pkl`), feature lists (`features.pkl`), and metadata json (`model_info.json`).
- `insurance.csv`: Primary medical dataset.
- `requirements.txt`: Python package dependencies.
- `.venv/`: Python virtual environment.

---

## 🛠️ How to Use

### 1. Setup Virtual Environment
It is highly recommended to use the included virtual environment (`.venv`) to ensure all executable scripts and dependencies are correctly resolved.

* **On Windows (PowerShell)**:
  ```powershell
  # Activate the virtual environment
  .venv\Scripts\Activate.ps1
  
  # Install dependencies (if not already installed)
  pip install -r requirements.txt
  ```
* **On macOS/Linux**:
  ```bash
  # Activate the virtual environment
  source .venv/bin/activate
  
  # Install dependencies
  pip install -r requirements.txt
  ```

### 2. Run Model Training & Analysis
Generate analysis plots and train the machine learning models:
```bash
python notebook/analysis.py
```

### 3. Launch the Streamlit App
Run the interactive dashboard locally:
```bash
streamlit run app.py
```
*(Alternatively, you can run it directly using the virtual environment interpreter path: `.venv/Scripts/python -m streamlit run app.py`)*

---

## 📊 Performance
The models were evaluated on an 80-20 train-test split:
- **Selected Best Model**: XGBoost
- **$R^2$ Score**: `0.8929`
- **Mean Absolute Error (MAE)**: `$2,437.28`

---
*Developed for health insurance analytics and premium prediction.*
