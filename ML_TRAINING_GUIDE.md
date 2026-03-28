# 📚 ML Model Training Guide

This guide covers training machine learning models for the Professional Trading Bot, including data preparation, feature engineering, model training, and MQL5 integration.

## 🎯 Training Overview

The bot uses an ensemble of three ML models:
1. **Random Forest** - Robust ensemble method
2. **XGBoost** - High-performance gradient boosting
3. **LSTM** - Deep learning for temporal patterns

## 📊 Data Preparation

### 1. Data Sources

#### MetaTrader 5 Historical Data
```python
import MetaTrader5 as mt5

# Connect to MT5
mt5.initialize()

# Get historical data
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
```

#### CSV Data Format
Expected columns:
```
timestamp, open, high, low, close, volume
2024-01-01 00:00:00, 1.0950, 1.0960, 1.0940, 1.0955, 1000
```

### 2. Data Quality Checks

```python
def validate_data(df):
    """Validate historical data quality"""
    assert not df.isnull().any().any(), "Missing values found"
    assert (df['high'] >= df['close']).all(), "High < Close"
    assert (df['low'] <= df['close']).all(), "Low > Close"
    assert (df['volume'] > 0).all(), "Zero volume"
    print("✅ Data validation passed")

validate_data(df)
```

## 🔧 Feature Engineering

### Technical Indicators

```python
from advanced_models.models import create_feature_engineering_pipeline

# Initialize pipeline
pipeline = create_feature_engineering_pipeline()

# Calculate indicators
df = pipeline["calculate_indicators"](df)

# View available features
print("Available features:")
print(df.columns.tolist())
```

### Custom Features

```python
def add_custom_features(df):
    """Add domain-specific features"""

    # Price momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

    # Volatility measures
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['gap_up'] = (df['open'] - df['close'].shift(1)) > 0
    df['gap_down'] = (df['open'] - df['close'].shift(1)) < 0

    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    return df

df = add_custom_features(df)
```

### Target Labels

```python
def create_labels(df, horizon=5, threshold=0.001):
    """
    Create prediction labels

    Args:
        horizon: Prediction horizon (periods ahead)
        threshold: Minimum return threshold
    """
    # Future return
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

    # Binary classification
    df['target'] = (df['future_return'] > threshold).astype(int)

    # Multi-class (optional)
    df['target_multi'] = pd.cut(df['future_return'],
                               bins=[-np.inf, -0.002, -0.001, 0.001, 0.002, np.inf],
                               labels=[0, 1, 2, 3, 4])

    return df.dropna()

df = create_labels(df, horizon=5)
```

## 🤖 Model Training

### 1. Data Splitting

```python
from sklearn.model_selection import train_test_split

# Feature columns
feature_cols = [col for col in df.columns if col not in ['target', 'future_return', 'timestamp']]

X = df[feature_cols].fillna(0).values
y = df['target'].values

# Time-series split (important for financial data)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

### 2. Random Forest Training

```python
from advanced_models.models import RandomForestPredictor

# Initialize model
rf_model = RandomForestPredictor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

# Train
metrics = rf_model.train(X_train, y_train)
print("Random Forest Results:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print("Classification Report:")
print(metrics['classification_report'])

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 features:")
print(feature_importance.head(10))
```

### 3. XGBoost Training

```python
from advanced_models.models import XGBoostPredictor

# Initialize model
xgb_model = XGBoostPredictor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Train
metrics = xgb_model.train(X_train, y_train)
print("XGBoost Results:")
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Feature importance plot
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model.model, max_num_features=20)
plt.show()
```

### 4. LSTM Training

```python
from advanced_models.models import LSTMPredictor

# Prepare sequential data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 60  # 60 periods lookback
X_seq = create_sequences(X_train, seq_length)
y_seq = y_train[seq_length:]

# Initialize model
lstm_model = LSTMPredictor(
    sequence_length=seq_length,
    n_features=X.shape[1],
    lstm_units=100,
    dropout_rate=0.2
)

# Train
metrics = lstm_model.train(X_seq, y_seq, epochs=50, batch_size=64)
print("LSTM Results:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### 5. Ensemble Training

```python
from ensemble.predictor import EnsemblePredictor

# Create ensemble
ensemble = EnsemblePredictor()

# Add trained models
ensemble.add_model('rf', rf_model)
ensemble.add_model('xgb', xgb_model)
ensemble.add_model('lstm', lstm_model)

# Test ensemble
test_features = X_test[:10]  # Test on first 10 samples
for i, features in enumerate(test_features):
    result = ensemble.predict(features.reshape(1, -1))
    print(f"Sample {i}: Prediction={result['prediction']}, "
          f"Confidence={result['confidence']:.3f}, "
          f"Regime={result['regime']}")
```

## 📈 Model Evaluation

### Performance Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""

    # Predictions
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
        if isinstance(y_pred, tuple):  # For models returning probabilities
            y_pred = y_pred[0]
    else:
        # For ensemble
        y_pred = []
        for x in X_test:
            result = model.predict(x.reshape(1, -1))
            y_pred.append(result['prediction'])
        y_pred = np.array(y_pred)

    # Classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Trading metrics
    trades = []
    for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
        if pred == 1:  # Long signal
            # Simulate trade (simplified)
            entry_price = 1.0  # Would use actual price
            exit_price = entry_price * (1.01 if actual == 1 else 0.99)
            profit = exit_price - entry_price
            trades.append(profit)

    if trades:
        win_rate = sum(1 for t in trades if t > 0) / len(trades)
        avg_win = sum(t for t in trades if t > 0) / max(1, sum(1 for t in trades if t > 0))
        avg_loss = sum(t for t in trades if t < 0) / max(1, sum(1 for t in trades if t < 0))

        print("
Trading Metrics:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Avg Win: ${avg_win:.4f}")
        print(f"Avg Loss: ${avg_loss:.4f}")
        print(f"Profit Factor: {abs(sum(t for t in trades if t > 0) / sum(t for t in trades if t < 0)):.2f}")

# Evaluate all models
evaluate_model(rf_model, X_test, y_test, "Random Forest")
evaluate_model(xgb_model, X_test, y_test, "XGBoost")
evaluate_model(ensemble, X_test, y_test, "Ensemble")
```

### Walk-Forward Validation

```python
def walk_forward_validation(X, y, model_class, window_size=1000, step_size=100):
    """
    Walk-forward validation for time series
    """
    scores = []

    for i in range(window_size, len(X), step_size):
        # Training window
        X_train_wf = X[i-window_size:i]
        y_train_wf = y[i-window_size:i]

        # Test window
        X_test_wf = X[i:i+step_size]
        y_test_wf = y[i:i+step_size]

        if len(X_test_wf) == 0:
            break

        # Train model
        model = model_class()
        model.train(X_train_wf, y_train_wf)

        # Test model
        y_pred = model.predict(X_test_wf)[0]
        accuracy = np.mean(y_pred == y_test_wf)
        scores.append(accuracy)

    return scores

# Run walk-forward validation
rf_scores = walk_forward_validation(X, y, RandomForestPredictor)
xgb_scores = walk_forward_validation(X, y, XGBoostPredictor)

print(f"RF WFV Mean Accuracy: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores):.4f})")
print(f"XGB WFV Mean Accuracy: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")
```

## 💾 Model Persistence

```python
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Save individual models
rf_model.save_model('models/rf_model.pkl')
xgb_model.save_model('models/xgb_model.pkl')
lstm_model.save_model('models/lstm_model.pkl')

# Save ensemble
ensemble.save_ensemble('models/')

print("✅ Models saved successfully")
```

## 🔄 Model Updating

### Online Learning

```python
def update_model_online(model, new_X, new_y, update_freq=100):
    """
    Update model with new data periodically
    """
    if len(new_X) >= update_freq:
        # Partial fit or retrain
        if hasattr(model.model, 'partial_fit'):
            model.model.partial_fit(new_X, new_y)
        else:
            # Retrain with recent data
            recent_X = new_X[-1000:]  # Last 1000 samples
            recent_y = new_y[-1000:]
            model.train(recent_X, recent_y)

        print("✅ Model updated with new data")
        return True

    return False
```

### Performance Monitoring

```python
def monitor_model_performance(predictions, actuals, window=100):
    """
    Monitor model performance over time
    """
    if len(predictions) < window:
        return None

    recent_preds = predictions[-window:]
    recent_actuals = actuals[-window:]

    accuracy = np.mean(recent_preds == recent_actuals)

    # Alert if performance drops
    if accuracy < 0.5:  # Below 50% accuracy
        print("⚠️ Model performance degraded!")

    return accuracy
```

## 🔗 MQL5 Integration

### 1. MQL5 EA Setup

The MQL5 Expert Advisor (`mql5/EA.mq5`) handles:
- Socket communication with Python
- Order execution
- Market data provision
- Position management

### 2. Socket Communication

```mql5
// MQL5 side - Send market data
void SendMarketData()
{
    string response = "{";
    response += "\"symbol\": \"" + Symbol() + "\",";
    response += "\"close\": " + DoubleToString(iClose(Symbol(), PERIOD_CURRENT, 0), 5) + ",";
    response += "\"volume\": " + IntegerToString(iVolume(Symbol(), PERIOD_CURRENT, 0));
    response += "}";
    
    SendResponse("{\"market_data\": " + response + "}");
}
```

### 3. Python-MQL5 Data Flow

```
Python Bot → Socket → MQL5 EA → MetaTrader 5 → Broker
    ↑                                       ↓
    ← Market Data ← Position Updates ← Order Confirmations
```

### 4. Testing MQL5 Integration

1. **Compile EA**: Load `EA.mq5` in MetaEditor
2. **Attach to Chart**: Attach EA to EURUSD chart
3. **Start Python Bot**: Run the trading bot
4. **Monitor Logs**: Check both MQL5 and Python logs

## 🚀 Deployment Checklist

- [ ] Historical data collected and validated
- [ ] Features engineered and tested
- [ ] Models trained and evaluated
- [ ] Walk-forward validation passed
- [ ] Ensemble weights optimized
- [ ] MQL5 EA compiled and tested
- [ ] Socket communication verified
- [ ] Risk management parameters set
- [ ] Backtesting completed
- [ ] Demo account testing (1-2 weeks)
- [ ] Live deployment with small lots

## ⚠️ Best Practices

1. **Data Quality**: Always validate your data
2. **Feature Selection**: Use domain knowledge for features
3. **Overfitting**: Use walk-forward validation
4. **Risk Management**: Never trade without stops
5. **Monitoring**: Continuously monitor performance
6. **Updates**: Regularly retrain models with new data

## 📞 Troubleshooting

### Common Issues

**Low Accuracy**:
- Check feature quality
- Try different model parameters
- Add more training data

**Overfitting**:
- Use regularization
- Implement walk-forward validation
- Simplify features

**MQL5 Connection Issues**:
- Check firewall settings
- Verify port 5000 is open
- Ensure EA is attached and running

---

**Happy Training! 🎯📊**