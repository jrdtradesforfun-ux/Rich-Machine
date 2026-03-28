#!/usr/bin/env python3
"""
Professional Trading Bot - Complete Implementation

A production-ready trading bot combining advanced machine learning,
risk management, broker integration, and real-time monitoring.

Features:
- Multi-symbol trading (EURUSD, GBPUSD, USDJPY)
- Ensemble ML predictions (Random Forest, XGBoost, LSTM)
- Market regime detection
- Advanced risk management
- Real-time performance monitoring
- Automated trade execution

Author: Professional Trading Bot
Date: 2024
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import signal
import sys

# Import bot components
from brokers.universal_broker import UniversalBroker
from execution.engine import ExecutionEngine, RiskManager
from monitoring.metrics import PerformanceMonitor
from ensemble.predictor import EnsemblePredictor
from advanced_models.models import create_feature_engineering_pipeline

class ProfessionalTradingBot:
    """
    Professional Trading Bot
    
    Main orchestrator for automated trading system.
    """
    
    def __init__(self, account_size: float = 10000, risk_per_trade: float = 0.02):
        """
        Initialize the trading bot
        
        Args:
            account_size: Account balance in USD
            risk_per_trade: Risk per trade as fraction (0.02 = 2%)
        """
        print("🤖 Initializing Professional Trading Bot...")
        
        # Configuration
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.trading_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        self.is_running = False
        self.last_prediction_time = {}
        
        # Initialize components
        self.broker = UniversalBroker()
        self.risk_manager = RiskManager(account_size, risk_per_trade)
        self.execution_engine = ExecutionEngine(self.broker, self.risk_manager)
        self.monitor = PerformanceMonitor()
        self.ensemble = EnsemblePredictor()
        
        # Market data storage
        self.market_data = {symbol: [] for symbol in self.trading_symbols}
        self.feature_pipeline = create_feature_engineering_pipeline()
        
        # Training data
        self.training_data = None
        
        print("✅ Bot initialized successfully")
    
    def connect(self) -> bool:
        """
        Connect to broker
        
        Returns:
            bool: True if connection successful
        """
        print("🔌 Connecting to broker...")
        if self.broker.connect():
            balance = self.broker.get_account_balance()
            print(f"✅ Connected. Account Balance: ${balance:.2f}")
            return True
        else:
            print("❌ Failed to connect to broker")
            return False
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        print("🧠 Loading/training ML models...")
        
        # Generate sample training data (in production, use real historical data)
        self.training_data = self._generate_sample_training_data()
        
        if self.training_data is None:
            print("❌ No training data available")
            return
            
        X, y = self.training_data
        
        # Train ensemble
        try:
            self.ensemble.train_ensemble(X, y)
            print("✅ Models trained successfully")
        except Exception as e:
            print(f"❌ Model training failed: {e}")
    
    def start(self, duration_minutes: int = 120):
        """
        Start the trading bot
        
        Args:
            duration_minutes: How long to run (0 for indefinite)
        """
        if not self.connect():
            return
            
        self.load_or_train_models()
        
        print(f"🚀 Starting trading bot for {duration_minutes} minutes...")
        print(f"📊 Trading symbols: {', '.join(self.trading_symbols)}")
        print(f"⚠️  Risk per trade: {self.risk_per_trade*100}%")
        
        self.is_running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60) if duration_minutes > 0 else float('inf')
        
        try:
            while self.is_running and time.time() < end_time:
                self._trading_loop()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        except Exception as e:
            print(f"\n❌ Trading loop error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        print("🛑 Stopping trading bot...")
        self.is_running = False
        
        # Close all positions
        closed = self.execution_engine.close_all_positions()
        print(f"📊 Closed {closed} positions")
        
        # Disconnect
        self.broker.disconnect()
        
        # Final report
        self._print_final_report()
        
        print("✅ Trading bot stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        for symbol in self.trading_symbols:
            try:
                # Get market data
                market_data = self.broker.get_market_data(symbol)
                if not market_data:
                    continue
                    
                # Update market history
                self._update_market_data(symbol, market_data)
                
                # Check if we should make a prediction
                if self._should_predict(symbol):
                    signal = self._generate_signal(symbol, market_data)
                    
                    if signal and signal['confidence'] > 0.6:  # Minimum confidence
                        result = self.execution_engine.execute_signal(signal)
                        
                        if result['success']:
                            print(f"✅ Executed {signal['direction']} {symbol} at {signal['entry_price']}")
                        else:
                            print(f"❌ Failed to execute {symbol}: {result.get('error', 'Unknown error')}")
                    
                    self.last_prediction_time[symbol] = time.time()
                    
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
    
    def _generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal using ML ensemble
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            Trading signal dictionary or None
        """
        try:
            # Get recent data for features
            recent_data = self._get_recent_data(symbol, 100)
            if len(recent_data) < 50:
                return None
                
            # Calculate features
            df = pd.DataFrame(recent_data)
            df = self.feature_pipeline["calculate_indicators"](df)
            
            # Get latest features
            latest_features = df.iloc[-1:].dropna()
            if latest_features.empty:
                return None
                
            feature_array = latest_features.values
            
            # Make prediction
            prediction_result = self.ensemble.predict(feature_array, market_data)
            
            if prediction_result['prediction'] == 0:
                return None  # No trade signal
                
            # Create signal
            current_price = market_data.get('close', market_data.get('bid', 0))
            direction = "long" if prediction_result['prediction'] == 1 else "short"
            
            # Calculate stop loss and take profit (simplified)
            atr = latest_features['ATR'].iloc[0] if 'ATR' in latest_features.columns else 0.001
            stop_distance = atr * 2  # 2 ATR stop
            target_distance = atr * 3  # 3 ATR target
            
            if direction == "long":
                stop_loss = current_price - stop_distance
                take_profit = current_price + target_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - target_distance
            
            signal = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": prediction_result['confidence'],
                "regime": prediction_result['regime'],
                "timestamp": datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            print(f"❌ Error generating signal for {symbol}: {e}")
            return None
    
    def _update_market_data(self, symbol: str, data: Dict[str, Any]):
        """Update market data history"""
        self.market_data[symbol].append({
            'timestamp': time.time(),
            'open': data.get('open', 0),
            'high': data.get('high', 0),
            'low': data.get('low', 0),
            'close': data.get('close', 0),
            'volume': data.get('volume', 0)
        })
        
        # Keep only recent data
        max_history = 1000
        if len(self.market_data[symbol]) > max_history:
            self.market_data[symbol] = self.market_data[symbol][-max_history:]
    
    def _get_recent_data(self, symbol: str, n: int) -> List[Dict]:
        """Get recent market data for symbol"""
        data = self.market_data[symbol]
        return data[-n:] if len(data) >= n else data
    
    def _should_predict(self, symbol: str) -> bool:
        """Check if we should make a prediction for this symbol"""
        last_time = self.last_prediction_time.get(symbol, 0)
        return time.time() - last_time > 300  # 5 minutes between predictions
    
    def _generate_sample_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate sample training data (replace with real data)"""
        try:
            # Generate synthetic OHLCV data
            np.random.seed(42)
            n_samples = 1000
            
            # Simulate price series
            prices = [1.1000]
            for i in range(n_samples - 1):
                change = np.random.normal(0, 0.001)
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            # Create OHLCV data
            data = []
            for i in range(len(prices) - 1):
                open_price = prices[i]
                close_price = prices[i + 1]
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.002))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.002))
                volume = np.random.randint(100, 1000)
                
                data.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            
            # Calculate features
            df = self.feature_pipeline["calculate_indicators"](df)
            
            # Create labels
            df = self.feature_pipeline["create_labels"](df)
            
            # Prepare X, y
            feature_cols = [col for col in df.columns if col not in ['target', 'future_return']]
            X = df[feature_cols].fillna(0).values
            y = df['target'].values
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error generating training data: {e}")
            return None
    
    def _print_final_report(self):
        """Print final trading report"""
        metrics = self.monitor.get_metrics()
        daily_report = self.monitor.get_daily_report()
        
        print("\n" + "="*50)
        print("📊 FINAL TRADING REPORT")
        print("="*50)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: ${metrics['max_drawdown']:.2f}")
        print(f"Running Time: {metrics['running_time']/3600:.1f} hours")
        print(f"Daily P&L: ${daily_report.get('profit', 0):.2f}")
        print("="*50)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutdown signal received")
        self.is_running = False

def main():
    """Main entry point"""
    print("🚀 Professional Trading Bot")
    print("⚠️  IMPORTANT: This is for educational purposes only")
    print("⚠️  Never risk real money without thorough testing")
    print()
    
    # Create bot instance
    bot = ProfessionalTradingBot(
        account_size=10000,      # $10,000 account
        risk_per_trade=0.02      # 2% risk per trade
    )
    
    # Configure trading
    bot.trading_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    # Start trading (2 hours for demo)
    try:
        bot.start(duration_minutes=120)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

if __name__ == "__main__":
    main()