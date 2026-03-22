import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class InsuranceModelTrainer:
    def __init__(self, data_path='insurance.csv', output_dir='model', plots_dir='notebook/plots'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        self.df = None
        self.encoders = {}
        self.features = []
        self.target = 'charges'
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_data(self):
        """Loads and performs initial data cleaning."""
        try:
            # Using a local variable for the initial load to assist static analysis tools
            loaded_data = pd.read_csv(self.data_path)
            if loaded_data is not None and not loaded_data.empty:
                self.df = loaded_data
                logger.info(f"Loaded primary data: {self.data_path}. Shape: {self.df.shape}")
            else:
                raise ValueError("Dataset is empty or could not be loaded.")
        except Exception as e:
            logger.error(f"Error accessing dataset: {e}")
            raise

        # Check for missing values after assignment
        df = self.df
        assert df is not None
        
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values. Removing them...")
            self.df = df.dropna()
        
        # Remove duplicates
        df = self.df
        assert df is not None
        dupes = df.duplicated().sum()
        if dupes > 0:
            logger.info(f"Removing {dupes} duplicate rows.")
            self.df = df.drop_duplicates()
        
        return self.df

    def exploratory_analysis(self):
        """Generates EDA plots."""
        df = self.df
        if df is None or df.empty:
            logger.error("No data loaded. Skipping EDA.")
            return

        logger.info("Generating analytical plots for dark theme...")
        plt.style.use('dark_background')
        
        # 1. Distribution of Target (Charges)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[self.target], kde=True, color='#00d4ff')
        plt.title('Distribution of Health Insurance Charges')
        plt.savefig(f"{self.plots_dir}/target_distribution.png", transparent=True)
        
        # 2. Pairplot
        sns.pairplot(df, hue='smoker', palette='bright')
        plt.savefig(f"{self.plots_dir}/pairplot.png", transparent=True)
        
        # 3. Correlation Heatmap
        temp_df = df.copy()
        for col in temp_df.select_dtypes('object').columns:
            temp_df[col] = LabelEncoder().fit_transform(temp_df[col])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Variable Correlation Heatmap')
        plt.savefig(f"{self.plots_dir}/correlation_heatmap.png", transparent=True)
        
        plt.close('all')

    def preprocess_features(self):
        """Feature engineering and encoding."""
        df = self.df
        if df is None or df.empty:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Preprocessing features...")
        X = df.drop(self.target, axis=1)
        y = df[self.target]
        
        X_encoded = X.copy()
        # Save LabelEncoders for deployment
        for col in ['sex', 'smoker', 'region']:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
                self.encoders[col] = le
        
        self.features = X_encoded.columns.tolist()
        return X_encoded, y

    def train_and_evaluate(self):
        """Trains multiple models and picks the best one."""
        X, y = self.preprocess_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.08, random_state=42)
        }
        
        best_r2 = -np.inf
        best_model = None
        best_name = ""
        metrics_summary = {}

        logger.info("Evaluating models...")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            metrics_summary[name] = {"Significance (R2)": float(r2), "Error Margin (MAE)": float(mae), "RMSE": float(rmse)}
            logger.info(f"Analytical Test: {name} - R2: {r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

        logger.info(f"Selected Best Model: {best_name} with R2: {best_r2:.4f}")
        
        # Save evaluation plots for best model
        self.save_model_plots(best_model, best_name, X_test, y_test)
        
        return best_model, best_name, metrics_summary

    def save_model_plots(self, model, name, X_test, y_test):
        """Generates diagnostics for the best model."""
        y_pred = model.predict(X_test)
        
        # Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'Actual vs Predicted Charges ({name})')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(f"{self.plots_dir}/best_model_performance.png", transparent=True)
        
        # Feature Importance if applicable
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = pd.Series(model.feature_importances_, index=self.features)
            importances.sort_values().plot(kind='barh', color='teal')
            plt.title(f'Key Drivers ({name})')
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/feature_importance.png", transparent=True)
        plt.close('all')

    def save_artifacts(self, model, name, metrics):
        """Saves models, encoders, and metrics to files."""
        joblib.dump(model, f"{self.output_dir}/best_insurance_model.pkl")
        joblib.dump(self.encoders, f"{self.output_dir}/encoders.pkl")
        joblib.dump(self.features, f"{self.output_dir}/features.pkl")
        
        info = {
            "best_model_name": name,
            "metrics": metrics[name],
            "all_metrics": metrics,
            "features": self.features,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(f"{self.output_dir}/model_info.json", 'w') as f:
            json.dump(info, f, indent=4)
        
        logger.info("All artifacts (model, encoders, info) saved successfully in 'model/' directory.")

def main():
    trainer = InsuranceModelTrainer()
    try:
        trainer.load_data()
        trainer.exploratory_analysis()
        best_model, best_name, metrics = trainer.train_and_evaluate()
        trainer.save_artifacts(best_model, best_name, metrics)
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")

if __name__ == "__main__":
    main()
