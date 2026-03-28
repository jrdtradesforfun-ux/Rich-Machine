#!/usr/bin/env python3
"""
ONNX Model Training and Export Script
=====================================

This script demonstrates how to train ML models and export them to ONNX format
for native inference in MetaTrader 5 Expert Advisors.

Features:
- Train Random Forest, XGBoost, and LSTM models
- Export models to ONNX format
- Feature engineering for forex trading
- Model validation and performance metrics
- Automated ONNX export pipeline

Usage:
    python onnx_training_pipeline.py --symbol EURUSD --model random_forest
    python onnx_training_pipeline.py --symbol GBPUSD --model xgboost
    python onnx_training_pipeline.py --symbol USDJPY --model lstm
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import onnxruntime as ort
import onnx
from onnx import numpy_helper

# Import our custom modules
sys.path.append(str(Path(__file__).parent))
from advanced_models.models import RandomForestPredictor, XGBoostPredictor, LSTMPredictor
from feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ONNXTrainingPipeline:
    """Pipeline for training and exporting ONNX models for MQL5 integration."""

    def __init__(self, symbol: str, model_type: str, data_path: str = None):
        self.symbol = symbol
        self.model_type = model_type
        self.data_path = data_path or f"data/{symbol}_historical.csv"
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()

        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'class': RandomForestPredictor
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'class': XGBoostPredictor
            },
            'lstm': {
                'units': 64,
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'class': LSTMPredictor
            }
        }

    def load_data(self) -> pd.DataFrame:
        """Load historical forex data."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        logger.info(f"Loaded {len(df)} records for {self.symbol}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and labels for training."""
        logger.info("Engineering features...")

        # Generate features
        features_df = self.feature_engineer.create_features(df)

        # Create labels (simplified: 1 for up, 0 for down based on future returns)
        future_returns = df['close'].shift(-5) / df['close'] - 1  # 5-period future return
        labels = (future_returns > 0.001).astype(int)  # 0.1% threshold

        # Remove NaN values
        valid_idx = ~(features_df.isna().any(axis=1) | labels.isna())
        features_df = features_df[valid_idx]
        labels = labels[valid_idx]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df.values, labels.values,
            test_size=0.2, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the specified model type."""
        logger.info(f"Training {self.model_type} model...")

        config = self.model_configs[self.model_type]
        model_class = config.pop('class')

        # Initialize model
        model = model_class(**config)

        # Train model
        model.train(X_train, y_train)

        logger.info("Model training completed")
        return model

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance."""
        logger.info("Evaluating model performance...")

        # Get predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        print("\n" + "="*50)
        print(f"MODEL EVALUATION - {self.model_type.upper()}")
        print("="*50)
        print(classification_report(y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        # Calculate additional metrics
        accuracy = np.mean(predictions == y_test)
        print(".4f")

        return accuracy

    def export_to_onnx(self, model, output_path: str):
        """Export trained model to ONNX format."""
        logger.info(f"Exporting model to ONNX format: {output_path}")

        # Create sample input for ONNX export
        sample_input = np.random.randn(1, 50).astype(np.float32)  # 50 features

        # Export to ONNX
        model.export_to_onnx(sample_input, output_path)

        # Verify ONNX model
        self.verify_onnx_model(output_path, sample_input)

        logger.info("ONNX export completed successfully")

    def verify_onnx_model(self, onnx_path: str, sample_input: np.ndarray):
        """Verify the exported ONNX model works correctly."""
        logger.info("Verifying ONNX model...")

        # Load ONNX model
        session = ort.InferenceSession(onnx_path)

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        onnx_output = session.run([output_name], {input_name: sample_input})

        logger.info(f"ONNX model verified - Input shape: {sample_input.shape}, Output shape: {onnx_output[0].shape}")

    def save_scaler(self, output_path: str):
        """Save the feature scaler for MQL5 use."""
        scaler_path = output_path.replace('.onnx', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    def run_pipeline(self):
        """Run the complete training and export pipeline."""
        try:
            # Load data
            df = self.load_data()

            # Prepare features
            X_train, X_test, y_train, y_test = self.prepare_features(df)

            # Train model
            model = self.train_model(X_train, y_train)

            # Evaluate model
            accuracy = self.evaluate_model(model, X_test, y_test)

            # Export to ONNX
            onnx_path = f"models/{self.symbol}_{self.model_type}.onnx"
            Path("models").mkdir(exist_ok=True)
            self.export_to_onnx(model, onnx_path)

            # Save scaler
            self.save_scaler(onnx_path)

            logger.info(f"✅ Pipeline completed successfully!")
            logger.info(f"📊 Model Accuracy: {accuracy:.4f}")
            logger.info(f"📁 ONNX Model: {onnx_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="ONNX Model Training Pipeline")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., EURUSD)")
    parser.add_argument("--model", required=True,
                       choices=['random_forest', 'xgboost', 'lstm'],
                       help="Model type to train")
    parser.add_argument("--data", help="Path to historical data CSV file")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ONNXTrainingPipeline(args.symbol, args.model, args.data)

    # Run pipeline
    success = pipeline.run_pipeline()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()