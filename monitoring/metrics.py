from typing import Dict, List, Any
import time
import statistics
from datetime import datetime, timedelta

class TradeRecord:
    """Record of a completed trade"""
    
    def __init__(self, entry_price: float, exit_price: float, direction: str, 
                 volume: float, symbol: str, entry_time: float, exit_time: float,
                 profit: float):
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.direction = direction
        self.volume = volume
        self.symbol = symbol
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.profit = profit
        self.duration = exit_time - entry_time

class PerformanceMonitor:
    """
    Real-time Performance Monitoring
    
    Tracks trading performance metrics, alerts, and system health.
    """
    
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.daily_stats = {}
        self.alerts = []
        self.start_time = time.time()
        
    def record_trade(self, entry_price: float, exit_price: float, direction: str,
                    volume: float, symbol: str, entry_time: float, exit_time: float,
                    profit: float):
        """
        Record a completed trade
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: long/short
            volume: Position size
            symbol: Trading symbol
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            profit: Profit/loss amount
        """
        trade = TradeRecord(entry_price, exit_price, direction, volume, 
                          symbol, entry_time, exit_time, profit)
        self.trades.append(trade)
        
        # Check for alerts
        self._check_trade_alerts(trade)
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate current performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return self._empty_metrics()
            
        profits = [t.profit for t in self.trades]
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit < 0]
        
        total_profit = sum(profits)
        win_rate = len(winning_trades) / len(self.trades)
        avg_win = statistics.mean([t.profit for t in winning_trades]) if winning_trades else 0
        avg_loss = statistics.mean([t.profit for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum([t.profit for t in winning_trades]) / 
                          sum([t.profit for t in losing_trades])) if losing_trades else float('inf')
        
        # Sharpe ratio (simplified)
        returns = profits
        if len(returns) > 1:
            sharpe = statistics.mean(returns) / statistics.stdev(returns) if statistics.stdev(returns) > 0 else 0
        else:
            sharpe = 0
            
        # Max drawdown (simplified)
        cumulative = [sum(profits[:i+1]) for i in range(len(profits))]
        peak = max(cumulative) if cumulative else 0
        trough = min(cumulative) if cumulative else 0
        max_drawdown = peak - trough
        
        return {
            "total_trades": len(self.trades),
            "total_profit": total_profit,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "running_time": time.time() - self.start_time
        }
    
    def get_daily_report(self) -> Dict[str, Any]:
        """Generate daily trading report"""
        today = datetime.now().date()
        today_trades = [t for t in self.trades 
                       if datetime.fromtimestamp(t.entry_time).date() == today]
        
        if not today_trades:
            return {"date": str(today), "trades": 0, "profit": 0}
            
        profit = sum([t.profit for t in today_trades])
        
        return {
            "date": str(today),
            "trades": len(today_trades),
            "profit": profit,
            "win_rate": len([t for t in today_trades if t.profit > 0]) / len(today_trades)
        }
    
    def add_alert(self, message: str, severity: str = "info"):
        """
        Add an alert
        
        Args:
            message: Alert message
            severity: info/warning/error
        """
        alert = {
            "timestamp": time.time(),
            "message": message,
            "severity": severity
        }
        self.alerts.append(alert)
        
        # In a real system, this would send notifications
        print(f"[{severity.upper()}] {message}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours"""
        cutoff = time.time() - (hours * 3600)
        return [a for a in self.alerts if a["timestamp"] > cutoff]
    
    def _check_trade_alerts(self, trade: TradeRecord):
        """Check for trade-related alerts"""
        if trade.profit < -100:  # Large loss
            self.add_alert(f"Large loss: ${trade.profit:.2f} on {trade.symbol}", "warning")
        elif trade.profit > 500:  # Large win
            self.add_alert(f"Large win: ${trade.profit:.2f} on {trade.symbol}", "info")
            
        # Daily loss limit check would be implemented here
        daily_profit = sum([t.profit for t in self.trades 
                          if datetime.fromtimestamp(t.entry_time).date() == datetime.now().date()])
        if daily_profit < -500:  # 5% of $10k account
            self.add_alert("Daily loss limit reached", "error")
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            "total_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "running_time": time.time() - self.start_time
        }