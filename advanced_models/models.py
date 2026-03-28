import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

# ONNX support
try:
    import tf2onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX libraries not available. Install: pip install tf2onnx onnxruntime")

class BasePredictor:
    """Base class for ML predictors with ONNX support"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.onnx_session = None
        self.onnx_path = None

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict and return probabilities"""
        raise NotImplementedError

    def save_model(self, path: str):
        """Save model to disk"""
        raise NotImplementedError

    def load_model(self, path: str):
        """Load model from disk"""
        raise NotImplementedError

    def export_to_onnx(self, output_path: str, input_sample: np.ndarray = None) -> bool:
        """
        Export model to ONNX format for MQL5 integration

        Args:
            output_path: Path to save ONNX model
            input_sample: Sample input for model conversion

        Returns:
            bool: True if export successful
        """
        if not ONNX_AVAILABLE:
            print("❌ ONNX libraries not available")
            return False

        try:
            if self.model_name == "lstm" and hasattr(self, '_export_lstm_to_onnx'):
                return self._export_lstm_to_onnx(output_path, input_sample)
            elif self.model_name in ["rf", "xgb"] and hasattr(self, '_export_tree_to_onnx'):
                return self._export_tree_to_onnx(output_path, input_sample)
            else:
                print(f"❌ ONNX export not supported for {self.model_name}")
                return False
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return False

    def load_onnx_model(self, onnx_path: str):
        """Load ONNX model for inference"""
        if not ONNX_AVAILABLE:
            print("❌ ONNX runtime not available")
            return False

        try:
            self.onnx_session = ort.InferenceSession(onnx_path)
            self.onnx_path = onnx_path
            print(f"✅ ONNX model loaded: {onnx_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False

    def predict_onnx(self, X: np.ndarray) -> np.ndarray:
        """Run inference using ONNX model"""
        if not self.onnx_session:
            raise ValueError("ONNX model not loaded")

        try:
            # Prepare input
            input_name = self.onnx_session.get_inputs()[0].name
            X_reshaped = X.astype(np.float32)

            # Run inference
            result = self.onnx_session.run(None, {input_name: X_reshaped})

            # Return prediction (assuming binary classification)
            if len(result) == 1:
                # Single output (probabilities)
                return result[0]
            else:
                # Multiple outputs (classes and probabilities)
                return result[1] if len(result) > 1 else result[0]

        except Exception as e:
            print(f"❌ ONNX inference failed: {e}")
            return np.array([])

class RandomForestPredictor(BasePredictor):
    """Random Forest Classifier for trading signals"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
        
    def save_model(self, path: str):
        """Save model"""
        joblib.dump(self.model, path)
        
    def load_model(self, path: str):
        """Load model"""
        self.model = joblib.load(path)
        self.is_trained = True

    def _export_tree_to_onnx(self, output_path: str, input_sample: np.ndarray = None) -> bool:
        """Export tree-based models (RF, XGB) to ONNX"""
        if input_sample is None:
            # Create sample input based on model
            n_features = self.model.n_features_in_
            input_sample = np.random.randn(1, n_features).astype(np.float32)

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Convert to ONNX
            initial_type = [('input', FloatTensorType([None, input_sample.shape[1]]))]
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            print(f"✅ ONNX model exported: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Tree ONNX export failed: {e}")
            return False

class XGBoostPredictor(BasePredictor):
    """XGBoost Classifier for trading signals"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__("xgboost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
        
    def save_model(self, path: str):
        """Save model"""
        self.model.save_model(path)
        
    def load_model(self, path: str):
        """Load model"""
        self.model.load_model(path)
        self.is_trained = True

class LSTMPredictor(BasePredictor):
    """LSTM Neural Network for trading signals"""
    
    def __init__(self, sequence_length: int = 60, n_features: int = 10, 
                 lstm_units: int = 50, dropout_rate: float = 0.2):
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.sequence_length, self.n_features), 
                 return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2),
            Dropout(self.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        return model
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train LSTM model"""
        # Reshape X for LSTM (samples, timesteps, features)
        if len(X.shape) == 2:
            # If 2D, assume we need to create sequences
            X_reshaped = self._create_sequences(X)
        else:
            X_reshaped = X
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=0.2, random_state=42
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        self.is_trained = True
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            "accuracy": accuracy,
            "loss": loss,
            "history": history.history
        }
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        # Reshape if needed
        if len(X.shape) == 2:
            X_reshaped = self._create_sequences(X)
        else:
            X_reshaped = X
            
        probabilities = self.model.predict(X_reshaped)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions, probabilities.flatten()
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input"""
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
        
    def save_model(self, path: str):
        """Save model"""
        self.model.save(path)
        
    def load_model(self, path: str):
        """Load model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        self.is_trained = True

    def _export_lstm_to_onnx(self, output_path: str, input_sample: np.ndarray = None) -> bool:
        """Export LSTM model to ONNX"""
        if input_sample is None:
            # Create sample input based on sequence length
            input_sample = np.random.randn(1, self.sequence_length, self.n_features).astype(np.float32)

        try:
            # Convert to ONNX using tf2onnx
            import tf2onnx

            # Get model spec
            spec = (tf.TensorSpec(input_sample.shape, tf.float32, name="input"),)

            # Convert model
            onnx_model, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=spec,
                opset=13,
                output_path=output_path
            )

            print(f"✅ LSTM ONNX model exported: {output_path}")
            return True

        except Exception as e:
            print(f"❌ LSTM ONNX export failed: {e}")
            return False

def create_feature_engineering_pipeline() -> Dict[str, Any]:
    """
    Create feature engineering pipeline for trading data
    
    Returns:
        Dictionary with feature engineering functions
    """
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        return df
    
    def create_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """Create prediction labels"""
        # Future return
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Label: 1 if price goes up, 0 if down
        df['target'] = (df['future_return'] > 0).astype(int)
        
        return df.dropna()
    
    return {
        "calculate_indicators": calculate_technical_indicators,
        "create_labels": create_labels
    }