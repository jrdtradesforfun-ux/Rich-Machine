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

class BasePredictor:
    """Base class for ML predictors"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
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