import socket
import json
import time
from typing import Dict, Any, Optional

class UniversalBroker:
    """
    Universal MT5 Broker Integration via Socket Connection

    Connects to MetaTrader 5 Expert Advisor running on the same machine
    via TCP socket for real-time trading operations.

    Works with any MT5 broker that supports the standard MQL5 API.
    """

    def __init__(self, host: str = "localhost", port: int = 5000, broker_name: str = "Universal"):
        self.host = host
        self.port = port
        self.broker_name = broker_name
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        """
        Establish connection to MT5 EA

        Returns:
            bool: True if connection successful
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✅ Connected to {self.broker_name} via MT5")
            return True
        except Exception as e:
            print(f"❌ Connection failed to {self.broker_name}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Close the socket connection"""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
        print(f"📴 Disconnected from {self.broker_name}")

    def _send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send command to MT5 and receive response

        Args:
            command: Command dictionary

        Returns:
            Response dictionary or None if failed
        """
        if not self.connected:
            return None

        try:
            # Send command
            data = json.dumps(command).encode('utf-8')
            self.socket.send(data)

            # Receive response
            response_data = self.socket.recv(4096)
            response = json.loads(response_data.decode('utf-8'))

            return response
        except Exception as e:
            print(f"❌ Command failed: {e}")
            return None

    def get_account_balance(self) -> float:
        """Get current account balance"""
        command = {"action": "get_balance"}
        response = self._send_command(command)
        return response.get("balance", 0.0) if response else 0.0

    def get_account_equity(self) -> float:
        """Get current account equity"""
        command = {"action": "get_equity"}
        response = self._send_command(command)
        return response.get("equity", 0.0) if response else 0.0

    def get_positions(self) -> list:
        """Get all open positions"""
        command = {"action": "get_positions"}
        response = self._send_command(command)
        return response.get("positions", []) if response else []

    def place_order(self, symbol: str, order_type: str, volume: float,
                   price: float = 0.0, sl: float = 0.0, tp: float = 0.0) -> bool:
        """
        Place a trading order

        Args:
            symbol: Trading symbol (EURUSD, etc.)
            order_type: buy/sell
            volume: Lot size
            price: Entry price (0 for market)
            sl: Stop loss price
            tp: Take profit price

        Returns:
            bool: True if order placed successfully
        """
        command = {
            "action": "place_order",
            "symbol": symbol,
            "type": order_type,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp
        }
        response = self._send_command(command)
        return response.get("success", False) if response else False

    def close_position(self, ticket: int) -> bool:
        """
        Close a position by ticket number

        Args:
            ticket: Position ticket

        Returns:
            bool: True if closed successfully
        """
        command = {"action": "close_position", "ticket": ticket}
        response = self._send_command(command)
        return response.get("success", False) if response else False

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information"""
        command = {"action": "get_symbol_info", "symbol": symbol}
        response = self._send_command(command)
        return response if response else {}

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        command = {"action": "get_market_data", "symbol": symbol}
        response = self._send_command(command)
        return response if response else {}