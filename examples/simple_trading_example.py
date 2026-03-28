#!/usr/bin/env python3
"""
Simple Trading Bot Example

A minimal example showing how to use the Professional Trading Bot
components for basic automated trading.
"""

import time
import numpy as np
from brokers.universal_broker import UniversalBroker
from execution.engine import ExecutionEngine, RiskManager
from monitoring.metrics import PerformanceMonitor
from ensemble.predictor import EnsemblePredictor
from advanced_models.models import RandomForestPredictor, XGBoostPredictor

def create_sample_bot():
    """Create a simple trading bot instance"""

    # Initialize components
    broker = UniversalBroker(host="localhost", port=5000)
    risk_manager = RiskManager(account_size=1000, risk_per_trade=0.01)  # 1% risk
    execution_engine = ExecutionEngine(broker, risk_manager)
    monitor = PerformanceMonitor()

    # Create ensemble with sample models
    ensemble = EnsemblePredictor()

    # Add basic models (would be trained in production)
    rf_model = RandomForestPredictor(n_estimators=50)
    xgb_model = XGBoostPredictor(n_estimators=50)

    # For demo, we'll use dummy trained models
    # In production: ensemble.add_model('rf', trained_rf_model)

    return {
        'broker': broker,
        'execution_engine': execution_engine,
        'monitor': monitor,
        'ensemble': ensemble
    }

def simple_trading_strategy(bot_components, symbol="EURUSD"):
    """
    Simple trend-following strategy example

    This is a basic example - replace with your ML predictions
    """

    broker = bot_components['broker']
    execution_engine = bot_components['execution_engine']
    monitor = bot_components['monitor']

    # Get market data
    market_data = broker.get_market_data(symbol)
    if not market_data:
        print(f"❌ No market data for {symbol}")
        return

    current_price = market_data.get('close', 0)
    if current_price == 0:
        return

    # Simple moving average crossover strategy (for demo)
    # In production, use your trained ML models

    # Simulate ML prediction (replace with real model)
    # For demo: random prediction with 55% win rate
    prediction = np.random.choice([0, 1], p=[0.45, 0.55])  # Slight bullish bias
    confidence = np.random.uniform(0.5, 0.8)

    if prediction == 1 and confidence > 0.6:  # Long signal
        direction = "long"
        stop_loss = current_price * 0.98  # 2% stop
        take_profit = current_price * 1.04  # 4% target
    elif prediction == 0 and confidence > 0.6:  # Short signal
        direction = "short"
        stop_loss = current_price * 1.02  # 2% stop
        take_profit = current_price * 0.96  # 4% target
    else:
        return  # No trade

    # Create signal
    signal = {
        "symbol": symbol,
        "direction": direction,
        "entry_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"📊 Signal generated: {direction.upper()} {symbol} at {current_price:.5f}")

    # Execute trade
    result = execution_engine.execute_signal(signal)

    if result['success']:
        print(f"✅ Trade executed - Ticket: {result.get('ticket', 'N/A')}")
    else:
        print(f"❌ Trade failed: {result.get('error', 'Unknown error')}")

def run_simple_bot(duration_minutes=5):
    """Run the simple trading bot"""

    print("🤖 Simple Trading Bot Demo")
    print("=" * 40)

    # Create bot
    bot = create_sample_bot()

    # Connect to broker
    if not bot['broker'].connect():
        print("❌ Failed to connect to broker. Make sure MQL5 EA is running.")
        return

    print("✅ Connected to broker")

    # Trading symbols
    symbols = ["EURUSD", "GBPUSD"]

    # Trading loop
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    print(f"🚀 Starting trading for {duration_minutes} minutes...")
    print("Press Ctrl+C to stop early")

    try:
        while time.time() < end_time:
            for symbol in symbols:
                simple_trading_strategy(bot, symbol)

            # Wait before next cycle
            time.sleep(30)  # Check every 30 seconds

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")

    # Cleanup
    bot['broker'].disconnect()

    # Show final report
    metrics = bot['monitor'].get_metrics()
    print("\n📊 Final Report:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Total P&L: ${metrics['total_profit']:.2f}")
    print(".1%")
    print("=" * 40)

if __name__ == "__main__":
    # Run demo for 2 minutes
    run_simple_bot(duration_minutes=2)