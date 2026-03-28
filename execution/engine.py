from typing import Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brokers.universal_broker import UniversalBroker
import time

class RiskManager:
    """
    Risk Management for Trade Execution
    
    Validates trades against risk parameters before execution.
    """
    
    def __init__(self, account_size: float = 10000, risk_per_trade: float = 0.02,
                 max_positions: int = 5, max_drawdown: float = 0.05):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade  # 2% per trade
        self.max_positions = max_positions
        self.max_drawdown = max_drawdown
        self.current_positions = 0
        self.initial_balance = account_size
        
    def validate_trade(self, signal: Dict[str, Any], broker: UniversalBroker) -> bool:
        """
        Validate trade against risk parameters
        
        Args:
            signal: Trading signal
            broker: Broker instance
            
        Returns:
            bool: True if trade passes validation
        """
        # Check position limit
        if self.current_positions >= self.max_positions:
            return False
            
        # Check drawdown
        current_equity = broker.get_account_equity()
        drawdown = (self.initial_balance - current_equity) / self.initial_balance
        if drawdown > self.max_drawdown:
            return False
            
        # Check margin
        symbol_info = broker.get_symbol_info(signal['symbol'])
        if not symbol_info:
            return False
            
        # Calculate position size based on risk
        risk_amount = self.account_size * self.risk_per_trade
        stop_loss_pips = abs(signal['entry_price'] - signal['stop_loss'])
        
        # Simple position sizing (can be enhanced)
        volume = risk_amount / (stop_loss_pips * symbol_info.get('point_value', 10))
        volume = min(volume, symbol_info.get('max_volume', 100))
        
        signal['volume'] = volume
        
        return True
    
    def update_positions(self, count: int):
        """Update current position count"""
        self.current_positions = count

class ExecutionEngine:
    """
    Trade Execution Engine
    
    Handles order placement, validation, and execution with risk management.
    """
    
    def __init__(self, broker: UniversalBroker, risk_manager: RiskManager):
        self.broker = broker
        self.risk_manager = risk_manager
        
    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal dictionary
                {
                    "symbol": "EURUSD",
                    "direction": "long",  # or "short"
                    "entry_price": 1.0950,
                    "stop_loss": 1.0900,
                    "take_profit": 1.1000,
                    "confidence": 0.75,
                    "timestamp": "2024-01-01T12:30:45"
                }
            
        Returns:
            Execution result dictionary
        """
        result = {
            "success": False,
            "ticket": None,
            "error": None,
            "timestamp": time.time()
        }
        
        # Validate signal
        if not self._validate_signal(signal):
            result["error"] = "Invalid signal format"
            return result
            
        # Risk validation
        if not self.risk_manager.validate_trade(signal, self.broker):
            result["error"] = "Risk validation failed"
            return result
            
        # Determine order type
        order_type = "buy" if signal["direction"] == "long" else "sell"
        
        # Execute order
        success = self.broker.place_order(
            symbol=signal["symbol"],
            order_type=order_type,
            volume=signal.get("volume", 0.01),
            price=signal.get("entry_price", 0.0),
            sl=signal.get("stop_loss", 0.0),
            tp=signal.get("take_profit", 0.0)
        )
        
        if success:
            result["success"] = True
            # Update position count
            positions = self.broker.get_positions()
            self.risk_manager.update_positions(len(positions))
        else:
            result["error"] = "Order placement failed"
            
        return result
    
    def close_position_by_ticket(self, ticket: int) -> bool:
        """Close position by ticket number"""
        return self.broker.close_position(ticket)
    
    def close_all_positions(self) -> int:
        """Close all open positions, return number closed"""
        positions = self.broker.get_positions()
        closed = 0
        for pos in positions:
            if self.broker.close_position(pos["ticket"]):
                closed += 1
        self.risk_manager.update_positions(0)
        return closed
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal format"""
        required_fields = ["symbol", "direction", "entry_price", "stop_loss", "take_profit"]
        for field in required_fields:
            if field not in signal:
                return False
        if signal["direction"] not in ["long", "short"]:
            return False
        return True