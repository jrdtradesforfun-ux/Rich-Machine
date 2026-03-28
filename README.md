# 🤖 Universal Professional Trading Bot

A complete, production-ready trading bot combining advanced machine learning, risk management, broker integration, execution pipeline, and real-time monitoring.

**Works with ANY MetaTrader 5 compatible broker**

## 🚀 Features

### Core Trading Capabilities
- ✅ **Multi-Symbol Trading** - Trade multiple currency pairs simultaneously (EURUSD, GBPUSD, USDJPY)
- ✅ **Real-time Market Data** - Live tick and bar data from MetaTrader 5
- ✅ **Smart Order Execution** - Automated order placement with risk validation
- ✅ **Position Management** - Open, close, and modify positions programmatically
- ✅ **Ensemble Predictions** - Multiple models (Random Forest, XGBoost, LSTM) with weighted voting
- ✅ **Market Regime Detection** - Automatically switches strategies based on trending/ranging/volatile markets
- ✅ **Confidence Scoring** - All predictions include confidence levels for trade filtering
- ✅ **Feature Engineering** - Technical indicators (MA, RSI, ATR, momentum)

### Risk Management
- ✅ **Risk Per Trade** - Configurable percentage-based position sizing
- ✅ **Stop Loss & Take Profit** - Automatic calculation based on price volatility
- ✅ **Position Limits** - Maximum concurrent positions validation
- ✅ **Margin Monitoring** - Real-time account margin checking
- ✅ **Drawdown Alerts** - Automatic trading halt if equity drops > 5%

### Monitoring & Alerting
- ✅ **Real-time Metrics** - Win rate, profit factor, Sharpe ratio, drawdown
- ✅ **Performance Alerts** - Automatic alerts on trades, equity changes, losses
- ✅ **System Health Monitoring** - Connection, data, and execution error tracking
- ✅ **Session Reports** - Daily trading summaries and statistics
- ✅ **Equity Tracking** - Real-time account balance and unrealized P&L

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Trading Bot (Your Machine or VPS)                       │
│  ├── ProfessionalTradingBot (Main)                              │
│  ├── EnsemblePredictor (ML voting)                              │
│  ├── ExecutionEngine (Order routing)                            │
│  ├── PerformanceMonitor (Metrics & alerts)                      │
│  └── SystemMonitor (Health tracking)                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Socket (Port 5000)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ MetaTrader 5 + Universal Bot                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Any MT5 Broker (JustMarkets, IC Markets, Pepperstone, etc.)    │
└──────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
/
├── brokers/                 # Broker connectors
│   └── universal_broker.py      # Universal MT5 integration
├── execution/              # Order execution pipeline
│   └── engine.py           # ExecutionEngine with validation
├── monitoring/             # Real-time monitoring
│   └── metrics.py          # Performance & system monitoring
├── advanced_models/        # ML predictors
│   └── models.py           # RF, XGBoost, LSTM models
├── ensemble/               # ML ensemble & regimes
│   └── predictor.py        # EnsemblePredictor & MarketRegimeDetector
├── trading_engine/         # Core trading logic
├── mql5_bridge/           # MT5 socket connector
├── data_processing/       # Data pipeline
└── pytrader/              # PyTrader API wrapper

examples/
├── professional_trading_bot.py    # Complete working example
└── example_*.py                   # Other examples

mql5/
└── EA.mq5     # MetaTrader 5 Expert Advisor
```

## 🎯 Quick Start

### 1. Prerequisites
- **MetaTrader 5** installed and running
- **Python 3.8+** installed
- **Any MT5-compatible** forex broker account

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/professional-trading-bot.git
cd professional-trading-bot

# Install Python dependencies
pip install -r requirements.txt

# Note: TensorFlow installation may take several minutes
```

### 3. MetaTrader 5 Setup
1. Open MetaTrader 5
2. Load the `mql5/EA.mq5` file as an Expert Advisor
3. Attach the EA to a chart (any symbol)
4. Ensure the EA is running (green play button)

### 4. Run the Bot

```bash
python examples/professional_trading_bot.py
```

## 🤖 Machine Learning Models

| Model | Type | Features |
|-------|------|----------|
| Random Forest | Ensemble Trees | Robust, handles noise well |
| XGBoost | Gradient Boosting | Fast, accurate predictions |
| LSTM | Deep Learning | Captures temporal patterns |

**Ensemble System**: Combines all 3 models with weighted voting for robust predictions.

## ⚙️ Core Components

### Broker Integration
```python
from .brokers import UniversalBroker

broker = UniversalBroker(host="localhost", port=5000)
broker.connect()
balance = broker.get_account_balance()
```

### Execution Engine
```python
from .execution import ExecutionEngine

engine = ExecutionEngine(broker, risk_manager)
signal = {"symbol": "EURUSD", "direction": "long", ...}
result = engine.execute_signal(signal)
```

### ML Ensemble
```python
from .ensemble import EnsemblePredictor

ensemble = EnsemblePredictor()
ensemble.add_model("rf", RandomForestPredictor())
ensemble.add_model("xgb", GradientBoostingPredictor())
prediction = ensemble.predict(features)
```

### Real-time Monitoring
```python
.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.record_trade(entry=1.0950, exit=1.1000, ...)
metrics = monitor.get_metrics()
```

## 📊 Configuration

### Risk Settings
```python
bot = ProfessionalTradingBot(
    account_size=10000,          # $10,000 account
    risk_per_trade=0.02,         # 2% per trade
)
```

### Trading Symbols
```python
bot.trading_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
```

### Duration
```python
bot.run(duration_minutes=120)  # 2 hours
```

## 📈 Trading Signals

Signal format:
```json
{
  "symbol": "EURUSD",
  "direction": "long",
  "entry_price": 1.0950,
  "stop_loss": 1.0900,
  "take_profit": 1.1000,
  "confidence": 0.75,
  "timestamp": "2024-01-01T12:30:45"
}
```

## 🚀 Deployment

### Local Testing
```bash
python examples/professional_trading_bot.py
```

### VPS (24/7 Automated)
```bash
# SSH into VPS, then:
python examples/professional_trading_bot.py &
# Or use screen/tmux for persistence
```

**Recommended VPS providers:**
- DigitalOcean ($5-6/mo)
- Contabo ($4-10/mo)

## 📚 Training Your ML Models

### 1. Data Collection
Collect historical forex data for your symbols:
- Use MetaTrader 5 historical data
- Download from brokers or data providers
- Ensure OHLCV data (Open, High, Low, Close, Volume)

### 2. Feature Engineering
The bot automatically calculates technical indicators:
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)

### 3. Model Training
```python
from advanced_models.models import create_feature_engineering_pipeline, RandomForestPredictor

# Load your historical data
# df = pd.read_csv('historical_data.csv')

# Create feature pipeline
pipeline = create_feature_engineering_pipeline()
df = pipeline["calculate_indicators"](df)
df = pipeline["create_labels"](df, horizon=5)  # 5-period ahead prediction

# Prepare training data
X = df[feature_columns].values
y = df['target'].values

# Train model
model = RandomForestPredictor()
model.train(X, y)

# Save model
model.save_model('models/rf_model.pkl')
```

### 4. Backtesting
- Test models on historical data
- Calculate Sharpe ratio, max drawdown
- Validate win rate and profit factor
- Ensure models generalize well

### 5. Live Trading
- Start with small position sizes
- Monitor performance closely
- Adjust models based on live results

## ⚠️ Risk Disclaimer

**IMPORTANT**: Forex trading with leverage is HIGH RISK.

**⚠️ START WITH DEMO ACCOUNT ⚠️**

Safe deployment practices:
- Test on demo 1-2 weeks minimum
- Backtest on historical data
- Start live with 0.01 lot positions
- Use stop losses on every trade
- Keep daily loss limit at 5%

## 📞 Support

For questions and support:
- Check the examples/ directory
- Review the code comments
- Test on demo account first

## 📄 License

This project is for educational purposes. Use at your own risk.

---

**Happy Trading! 🚀📈**