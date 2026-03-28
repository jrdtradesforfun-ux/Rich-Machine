# Universal ONNX ML Trading Bot

A professional-grade, enterprise-ready Python + MQL5 ML trading system with native ONNX model inference, Smart Money Concepts (SMC) strategies, and multi-timeframe analysis.

## 🚀 Features

### Core Capabilities
- **Native ONNX Inference**: Direct ML model execution in MQL5 without Python overhead
- **Multi-Model Support**: Random Forest, XGBoost, and LSTM models
- **SMC Trading Strategies**: Smart Money Concepts with fractal patterns and order blocks
- **Multi-Timeframe Analysis**: H4 context + M15 entry signals
- **Advanced Risk Management**: ATR-based stops, Kelly Criterion, drawdown limits
- **Kill Zone Filtering**: Trade only during optimal market hours
- **Real-time Monitoring**: Performance metrics and automated alerts

### Technical Specifications
- **Prediction Latency**: < 5ms (ONNX native inference)
- **Feature Engineering**: 50+ technical indicators
- **Risk Controls**: 2% max risk per trade, 5% daily drawdown limit
- **Position Limits**: Max 3 concurrent positions
- **Retry Logic**: 3-tier error handling with exponential backoff

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+
- **MetaTrader 5**: Build 3600+
- **RAM**: 8GB minimum
- **Storage**: 10GB for models and data

### Python Dependencies
```bash
pip install -r requirements.txt
```

### MQL5 Requirements
- Enable automated trading in MT5
- Allow DLL imports
- WebRequest access for data feeds

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python Bot    │    │     ONNX        │    │   MetaTrader 5  │
│                 │    │   Inference     │    │                 │
│ • Model Training│    │ • Native MQL5   │    │ • Order Execution│
│ • Feature Eng.  │◄──►│ • <5ms Latency  │◄──►│ • Risk Management│
│ • ONNX Export   │    │ • No Python Dep │    │ • SMC Strategies │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/jrdtradesforfun-ux/Rich-Machine.git
cd Rich-Machine
pip install -r requirements.txt
```

### 2. Train ONNX Model
```bash
# Train Random Forest model for EURUSD
python onnx_training_pipeline.py --symbol EURUSD --model random_forest

# Train XGBoost model for GBPUSD
python onnx_training_pipeline.py --symbol GBPUSD --model xgboost

# Train LSTM model for USDJPY
python onnx_training_pipeline.py --symbol USDJPY --model lstm
```

### 3. Deploy MQL5 EA
1. Copy `mql5/EA.mq5` to your MT5 Experts folder
2. Copy trained ONNX model to MT5 Files folder
3. Configure EA parameters in MT5
4. Enable automated trading

### 4. Start Trading
```bash
# Optional: Run Python monitoring bot
python examples/professional_trading_bot.py
```

## 📊 Model Training

### Supported Models

#### Random Forest
```bash
python onnx_training_pipeline.py --symbol EURUSD --model random_forest
```
- **Pros**: Fast training, interpretable, handles missing data
- **Use Case**: High-frequency signals, feature importance analysis

#### XGBoost
```bash
python onnx_training_pipeline.py --symbol GBPUSD --model xgboost
```
- **Pros**: High accuracy, handles complex patterns, gradient boosting
- **Use Case**: Trend following, breakout detection

#### LSTM (Neural Network)
```bash
python onnx_training_pipeline.py --symbol USDJPY --model lstm
```
- **Pros**: Time series patterns, memory of past sequences
- **Use Case**: Momentum trading, pattern recognition

### Training Data Format
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0850,1.0860,1.0845,1.0855,1250
2024-01-01 00:15:00,1.0855,1.0870,1.0850,1.0865,1180
...
```

## ⚙️ Configuration

### MQL5 EA Parameters

#### ONNX Configuration
```mq5
input string ONNX_Model_Path = "EURUSD_random_forest.onnx";
input int Feature_Count = 50;
input double Prediction_Threshold = 0.6;
```

#### SMC Strategy Settings
```mq5
input int H4_Timeframe = PERIOD_H4;
input int M15_Timeframe = PERIOD_M15;
input int Fractal_Period = 5;
```

#### Risk Management
```mq5
input double Risk_Per_Trade = 0.02;      // 2% risk per trade
input double Daily_Drawdown_Limit = 0.05; // 5% max daily drawdown
input int Max_Positions = 3;             // Max concurrent positions
```

#### Kill Zone Settings (GMT)
```mq5
input int London_Open_Hour = 8;          // London session start
input int London_Close_Hour = 10;        // London session end
input int NewYork_Open_Hour = 13;        // New York session start
input int NewYork_Close_Hour = 15;       // New York session end
```

### Python Configuration
```python
# Model training parameters
model_configs = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
    },
    'lstm': {
        'units': 64,
        'dropout': 0.2,
        'epochs': 50,
    }
}
```

## 📈 SMC Trading Strategies

### 1. Fractal Patterns (Liquidity Sweeps)
- Identifies institutional order flow
- 5-bar fractal analysis for entry confirmation
- Bullish/bearish fractal detection

### 2. Order Blocks
- Supply/demand zone identification
- Large candle followed by small candle pattern
- Institutional accumulation/distribution

### 3. Fibonacci Zones
- Premium zone (above H4 midpoint) - bearish bias
- Discount zone (below H4 midpoint) - bullish bias
- H4 context for M15 entries

### 4. Kill Zone Filtering
- London session: 8:00-10:00 GMT
- New York session: 13:00-15:00 GMT
- Optimal liquidity and volatility

## 🔧 Advanced Features

### ONNX Model Export
```python
# Export trained model to ONNX
model.export_to_onnx(sample_input, "model.onnx")

# Verify ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### Feature Engineering
```python
from feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features(price_data)

# 50+ features including:
# - Price action (OHLC, ranges, bodies)
# - Moving averages (SMA, EMA)
# - Oscillators (RSI, MACD, Stochastic)
# - Volatility (ATR, Bollinger Bands)
# - Volume indicators
# - Higher timeframe context
```

### Risk Management
```python
# ATR-based position sizing
atr = iATR(Symbol(), M15_Timeframe, 14, 0)
stop_loss = atr * 2  # 2 ATR stop
take_profit = atr * 3  # 3 ATR target

# Kelly Criterion integration
risk_amount = account_balance * risk_per_trade
position_size = risk_amount / (stop_pips * tick_value)
```

## 📊 Monitoring & Analytics

### Performance Metrics
- Win rate, profit factor, Sharpe ratio
- Maximum drawdown, recovery factor
- Trade frequency and holding time
- Risk-adjusted returns

### Real-time Alerts
- Telegram integration for trade signals
- Email alerts for risk breaches
- Performance dashboard updates

### Logging
```python
# Comprehensive logging
logger.info(f"Trade executed: {symbol} {order_type} {lot_size} lots")
logger.warning(f"Risk limit exceeded: {daily_drawdown:.2%}")
logger.error(f"ONNX inference failed: {error_message}")
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/ -v
```

### Integration Tests
```bash
python -m pytest tests/test_integration.py -v
```

### Performance Benchmarks
```bash
python benchmarks/model_latency.py
```

## 🚨 Risk Management

### Trade Filters
- ✅ Maximum 3 concurrent positions
- ✅ 2% risk per trade maximum
- ✅ 5% daily drawdown limit
- ✅ Kill zone time filtering
- ✅ SMC strategy confirmation

### Error Handling
- 3-tier retry logic with exponential backoff
- Circuit breaker for consecutive failures
- Graceful degradation to manual mode

### Validation Checks
- Pre-trade risk assessment
- Margin requirements verification
- Spread and slippage monitoring
- Broker connectivity checks

## 📚 Documentation

### User Guides
- [ML Training Guide](ML_TRAINING_GUIDE.md) - Model training and optimization
- [MQL5 Integration](MQL5_INTEGRATION.md) - EA deployment and configuration
- [Risk Management](RISK_MANAGEMENT.md) - Risk controls and monitoring

### API Reference
- [Python API](docs/python_api.md) - Complete Python module documentation
- [MQL5 API](docs/mql5_api.md) - Expert Advisor function reference

### Troubleshooting
- [Common Issues](docs/troubleshooting.md) - FAQ and solutions
- [Performance Tuning](docs/performance.md) - Optimization guides

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/your-username/Rich-Machine.git
cd Rich-Machine
pip install -r requirements-dev.txt
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jrdtradesforfun-ux/Rich-Machine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jrdtradesforfun-ux/Rich-Machine/discussions)
- **Documentation**: [Wiki](https://github.com/jrdtradesforfun-ux/Rich-Machine/wiki)

---

**Built with ❤️ for the algorithmic trading community**
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