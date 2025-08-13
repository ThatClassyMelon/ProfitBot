"""
Trade execution engine that processes trading signals and updates portfolio.
"""
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.strategy import TradeSignal, TradeAction
from utils.math_tools import format_currency, round_to_precision


class Trade:
    """Represents an executed trade."""
    
    def __init__(self, timestamp: datetime, coin: str, action: str, quantity: float, 
                 price: float, total_value: float, reason: str):
        self.timestamp = timestamp
        self.coin = coin
        self.action = action
        self.quantity = quantity
        self.price = price
        self.total_value = total_value
        self.reason = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for CSV export."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'coin': self.coin,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'total_value': self.total_value,
            'reason': self.reason
        }


class TradeExecutor:
    """Executes trades and manages trade history."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trade executor.
        
        Args:
            config: Configuration dictionary
        """
        self.trade_history_file = config['logging']['trade_history_file']
        self.trades: List[Trade] = []
        self._ensure_data_directory()
        self._initialize_trade_history_file()
    
    def _ensure_data_directory(self) -> None:
        """Ensure the data directory exists."""
        data_dir = os.path.dirname(self.trade_history_file)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def _initialize_trade_history_file(self) -> None:
        """Initialize the trade history CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.trade_history_file):
            headers = ['timestamp', 'coin', 'action', 'quantity', 'price', 'total_value', 'reason']
            with open(self.trade_history_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
    
    def execute_signals(self, signals: List[TradeSignal], portfolio) -> List[Trade]:
        """
        Execute a list of trade signals.
        
        Args:
            signals: List of trade signals to execute
            portfolio: Portfolio instance
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for signal in signals:
            if signal.action in [TradeAction.BUY, TradeAction.SELL]:
                trade = self._execute_signal(signal, portfolio)
                if trade:
                    executed_trades.append(trade)
                    self.trades.append(trade)
                    self._log_trade_to_csv(trade)
        
        return executed_trades
    
    def _execute_signal(self, signal: TradeSignal, portfolio) -> Optional[Trade]:
        """
        Execute a single trade signal.
        
        Args:
            signal: Trade signal to execute
            portfolio: Portfolio instance
            
        Returns:
            Trade object if successful, None otherwise
        """
        timestamp = datetime.now()
        
        if signal.action == TradeAction.BUY:
            success = portfolio.execute_buy(signal.coin, signal.quantity, signal.price)
        elif signal.action == TradeAction.SELL:
            success = portfolio.execute_sell(signal.coin, signal.quantity, signal.price)
        else:
            return None
        
        if success:
            total_value = signal.quantity * signal.price
            trade = Trade(
                timestamp=timestamp,
                coin=signal.coin,
                action=signal.action.value,
                quantity=signal.quantity,
                price=signal.price,
                total_value=total_value,
                reason=signal.reason
            )
            return trade
        
        return None
    
    def _log_trade_to_csv(self, trade: Trade) -> None:
        """
        Log a trade to the CSV file.
        
        Args:
            trade: Trade to log
        """
        try:
            with open(self.trade_history_file, 'a', newline='') as file:
                writer = csv.writer(file)
                trade_dict = trade.to_dict()
                writer.writerow([
                    trade_dict['timestamp'],
                    trade_dict['coin'],
                    trade_dict['action'],
                    trade_dict['quantity'],
                    trade_dict['price'],
                    trade_dict['total_value'],
                    trade_dict['reason']
                ])
        except Exception as e:
            print(f"Error logging trade to CSV: {e}")
    
    def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """
        Get recent trades.
        
        Args:
            limit: Number of recent trades to return
            
        Returns:
            List of recent trades
        """
        return self.trades[-limit:] if len(self.trades) > limit else self.trades
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get trading statistics.
        
        Returns:
            Dictionary of trading statistics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_volume': 0.0,
                'coins_traded': []
            }
        
        buy_trades = [t for t in self.trades if t.action == 'BUY']
        sell_trades = [t for t in self.trades if t.action == 'SELL']
        total_volume = sum(t.total_value for t in self.trades)
        coins_traded = list(set(t.coin for t in self.trades))
        
        return {
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_volume': round_to_precision(total_volume, 2),
            'coins_traded': coins_traded,
            'avg_trade_value': round_to_precision(total_volume / len(self.trades), 2) if self.trades else 0.0
        }
    
    def format_trade_summary(self, trades: List[Trade]) -> str:
        """
        Format a list of trades into a readable summary.
        
        Args:
            trades: List of trades to format
            
        Returns:
            Formatted trade summary string
        """
        if not trades:
            return "No trades executed this cycle."
        
        summary_lines = []
        for trade in trades:
            action_symbol = "📈" if trade.action == "BUY" else "📉"
            summary_lines.append(
                f"{action_symbol} {trade.action} {trade.quantity:.6f} {trade.coin} "
                f"@ ${trade.price:.2f} = {format_currency(trade.total_value)} "
                f"({trade.reason})"
            )
        
        return "\n".join(summary_lines)
    
    def clear_trade_history(self) -> None:
        """Clear all trade history."""
        self.trades.clear()
        # Recreate the CSV file with just headers
        self._initialize_trade_history_file()