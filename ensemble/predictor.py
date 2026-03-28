import numpy as np
from typing import Dict, List, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_models.models import BasePredictor, RandomForestPredictor, XGBoostPredictor, LSTMPredictor

class MarketRegimeDetector:
    """
    Market Regime Detection
    
    Detects trending, ranging, or volatile market conditions
    to switch trading strategies accordingly.
    """
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        
    def detect_regime(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        """
        Detect current market regime
        
        Args:
            prices: Array of closing prices
            volumes: Array of volumes
            
        Returns:
            Regime: 'trending', 'ranging', 'volatile'
        """
        if len(prices) < self.lookback_period:
            return 'unknown'
            
        recent_prices = prices[-self.lookback_period:]
        recent_volumes = volumes[-self.lookback_period:]
        
        # Calculate trend strength (slope of linear regression)
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        trend_strength = abs(slope) / np.mean(recent_prices)
        
        # Calculate volatility (standard deviation)
        returns = np.diff(np.log(recent_prices))
        volatility = np.std(returns)
        
        # Calculate volume trend
        volume_trend = np.polyfit(x, recent_volumes, 1)[0]
        
        # Classify regime
        if trend_strength > 0.001 and volatility < 0.02:
            return 'trending'
        elif volatility > 0.03:
            return 'volatile'
        else:
            return 'ranging'
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get model weights based on market regime
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of model weights
        """
        if regime == 'trending':
            return {'rf': 0.4, 'xgb': 0.4, 'lstm': 0.2}
        elif regime == 'ranging':
            return {'rf': 0.3, 'xgb': 0.3, 'lstm': 0.4}
        elif regime == 'volatile':
            return {'rf': 0.5, 'xgb': 0.3, 'lstm': 0.2}
        else:
            return {'rf': 0.33, 'xgb': 0.33, 'lstm': 0.34}

class EnsemblePredictor:
    """
    ML Ensemble Predictor
    
    Combines multiple ML models with weighted voting
    and market regime adaptation.
    """
    
    def __init__(self):
        self.models: Dict[str, BasePredictor] = {}
        self.regime_detector = MarketRegimeDetector()
        self.market_data_history = {'prices': [], 'volumes': []}
        
    def add_model(self, name: str, model: BasePredictor):
        """
        Add a model to the ensemble
        
        Args:
            name: Model name/key
            model: Trained predictor instance
        """
        self.models[name] = model
        
    def predict(self, features: np.ndarray, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make ensemble prediction
        
        Args:
            features: Feature array for prediction
            market_data: Current market data for regime detection
            
        Returns:
            Prediction result with confidence
        """
        if not self.models:
            raise ValueError("No models added to ensemble")
            
        # Update market data history for regime detection
        if market_data:
            self._update_market_history(market_data)
            
        # Detect market regime
        regime = self._detect_current_regime()
        weights = self.regime_detector.get_regime_weights(regime)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        confidences = {}
        
        for name, model in self.models.items():
            if name in weights:
                try:
                    pred, prob = model.predict(features.reshape(1, -1))
                    predictions[name] = pred[0]
                    probabilities[name] = prob
                    
                    # Calculate confidence (max probability)
                    if len(prob.shape) > 1:
                        confidences[name] = np.max(prob, axis=1)[0]
                    else:
                        confidences[name] = max(prob)
                        
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
                    predictions[name] = 0
                    probabilities[name] = np.array([0.5])
                    confidences[name] = 0.5
        
        # Weighted ensemble prediction
        weighted_prediction = self._weighted_vote(predictions, weights)
        ensemble_confidence = self._calculate_ensemble_confidence(confidences, weights)
        
        # Check for disagreement
        disagreement = self._check_disagreement(predictions)
        
        return {
            "prediction": weighted_prediction,
            "confidence": ensemble_confidence,
            "regime": regime,
            "individual_predictions": predictions,
            "individual_confidences": confidences,
            "disagreement": disagreement,
            "weights": weights
        }
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, model_configs: Dict[str, Dict] = None):
        """
        Train all models in the ensemble
        
        Args:
            X: Feature matrix
            y: Target labels
            model_configs: Configuration for each model
        """
        if model_configs is None:
            model_configs = {
                'rf': {'n_estimators': 100, 'max_depth': 10},
                'xgb': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
                'lstm': {'sequence_length': 60, 'n_features': X.shape[1], 'lstm_units': 50}
            }
        
        # Train Random Forest
        if 'rf' not in self.models:
            rf = RandomForestPredictor(**model_configs.get('rf', {}))
            rf.train(X, y)
            self.add_model('rf', rf)
            
        # Train XGBoost
        if 'xgb' not in self.models:
            xgb_model = XGBoostPredictor(**model_configs.get('xgb', {}))
            xgb_model.train(X, y)
            self.add_model('xgb', xgb_model)
            
        # Train LSTM (reshape data if needed)
        if 'lstm' not in self.models:
            lstm = LSTMPredictor(**model_configs.get('lstm', {}))
            lstm.train(X, y)
            self.add_model('lstm', lstm)
    
    def save_ensemble(self, directory: str):
        """Save all models in the ensemble"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}_model.pkl")
            model.save_model(path)
    
    def load_ensemble(self, directory: str):
        """Load all models from directory"""
        import os
        
        for name in ['rf', 'xgb', 'lstm']:
            path = os.path.join(directory, f"{name}_model.pkl")
            if os.path.exists(path):
                if name == 'rf':
                    model = RandomForestPredictor()
                elif name == 'xgb':
                    model = XGBoostPredictor()
                elif name == 'lstm':
                    model = LSTMPredictor()
                    
                model.load_model(path)
                self.add_model(name, model)
    
    def _weighted_vote(self, predictions: Dict[str, int], weights: Dict[str, float]) -> int:
        """Calculate weighted vote"""
        weighted_sum = 0
        total_weight = 0
        
        for name, pred in predictions.items():
            if name in weights:
                weighted_sum += pred * weights[name]
                total_weight += weights[name]
        
        return 1 if weighted_sum / total_weight > 0.5 else 0
    
    def _calculate_ensemble_confidence(self, confidences: Dict[str, float], 
                                     weights: Dict[str, float]) -> float:
        """Calculate ensemble confidence"""
        weighted_confidence = 0
        total_weight = 0
        
        for name, conf in confidences.items():
            if name in weights:
                weighted_confidence += conf * weights[name]
                total_weight += weights[name]
        
        return weighted_confidence / total_weight if total_weight > 0 else 0
    
    def _check_disagreement(self, predictions: Dict[str, int]) -> bool:
        """Check if models disagree significantly"""
        preds = list(predictions.values())
        if len(preds) < 2:
            return False
            
        # If more than half disagree with majority
        majority = max(set(preds), key=preds.count)
        disagreements = sum(1 for p in preds if p != majority)
        
        return disagreements > len(preds) / 2
    
    def _update_market_history(self, market_data: Dict[str, Any]):
        """Update market data history"""
        if 'close' in market_data:
            self.market_data_history['prices'].append(market_data['close'])
        if 'volume' in market_data:
            self.market_data_history['volumes'].append(market_data['volume'])
            
        # Keep only recent history
        max_history = 1000
        for key in self.market_data_history:
            if len(self.market_data_history[key]) > max_history:
                self.market_data_history[key] = self.market_data_history[key][-max_history:]
    
    def _detect_current_regime(self) -> str:
        """Detect current market regime"""
        prices = np.array(self.market_data_history['prices'])
        volumes = np.array(self.market_data_history['volumes'])
        
        if len(prices) == 0:
            return 'unknown'
            
        return self.regime_detector.detect_regime(prices, volumes)