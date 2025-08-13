"""
Enhanced trade execution engine with profit rebalancing support.
"""
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.enhanced_strategy import TradeSignal, TradeAction
from utils.math_tools import format_currency, round_to_precision


class EnhancedTrade:
    """Represents an executed trade with enhanced metadata."""
    
    def __init__(self, timestamp: datetime, coin: str, action: str, quantity: float, 
                 price: float, total_value: float, reason: str, tier: int = 0):
        self.timestamp = timestamp
        self.coin = coin
        self.action = action
        self.quantity = quantity
        self.price = price
        self.total_value = total_value
        self.reason = reason
        self.tier = tier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for CSV export."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'coin': self.coin,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'total_value': self.total_value,
            'reason': self.reason,
            'tier': self.tier
        }


class EnhancedTradeExecutor:
    """Enhanced trade executor with rebalancing and DCA tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced trade executor."""
        self.trade_history_file = config['logging']['trade_history_file']
        self.trades: List[EnhancedTrade] = []
        self.rebalance_trades: List[EnhancedTrade] = []
        self._ensure_data_directory()
        self._initialize_trade_history_file()
    
    def _ensure_data_directory(self) -> None:
        """Ensure the data directory exists."""
        data_dir = os.path.dirname(self.trade_history_file)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def _initialize_trade_history_file(self) -> None:
        """Initialize the trade history CSV file with enhanced headers."""
        if not os.path.exists(self.trade_history_file):
            headers = ['timestamp', 'coin', 'action', 'quantity', 'price', 'total_value', 'reason', 'tier']
            with open(self.trade_history_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
    
    def execute_signals(self, signals: List[TradeSignal], portfolio, strategy) -> List[EnhancedTrade]:
        """
        Execute a list of trade signals with enhanced logic.
        
        Args:
            signals: List of trade signals to execute
            portfolio: Portfolio instance
            strategy: Strategy instance for callbacks
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for signal in signals:
            if signal.action == TradeAction.REBALANCE:
                rebalance_trades = self._execute_rebalance(signal, portfolio)
                executed_trades.extend(rebalance_trades)
                for trade in rebalance_trades:
                    self.trades.append(trade)
                    self.rebalance_trades.append(trade)
                    self._log_trade_to_csv(trade)
                    
            elif signal.action in [TradeAction.BUY, TradeAction.SELL]:
                trade = self._execute_signal(signal, portfolio)
                if trade:
                    executed_trades.append(trade)
                    self.trades.append(trade)
                    self._log_trade_to_csv(trade)
                    
                    # Notify strategy of successful trade
                    strategy.on_trade_executed(signal)
        
        return executed_trades
    
    def _execute_signal(self, signal: TradeSignal, portfolio) -> Optional[EnhancedTrade]:
        """Execute a single trade signal."""
        timestamp = datetime.now()
        
        if signal.action == TradeAction.BUY:
            success = portfolio.execute_buy(signal.coin, signal.quantity, signal.price)
        elif signal.action == TradeAction.SELL:
            success = portfolio.execute_sell(signal.coin, signal.quantity, signal.price)
        else:
            return None
        
        if success:
            total_value = signal.quantity * signal.price
            trade = EnhancedTrade(
                timestamp=timestamp,
                coin=signal.coin,
                action=signal.action.value,
                quantity=signal.quantity,
                price=signal.price,
                total_value=total_value,
                reason=signal.reason,
                tier=signal.tier
            )
            return trade
        
        return None
    
    def _execute_rebalance(self, signal: TradeSignal, portfolio) -> List[EnhancedTrade]:
        """
        Execute profit rebalancing by selling percentage of all holdings.
        
        Args:
            signal: Rebalance signal
            portfolio: Portfolio instance
            
        Returns:
            List of rebalance trades executed
        """
        rebalance_trades = []
        timestamp = datetime.now()
        rebalance_percentage = signal.quantity  # Using quantity field for percentage
        
        holdings = portfolio.get_holdings()
        
        for coin, quantity in holdings.items():
            if quantity <= 0:
                continue
            
            # Calculate amount to sell
            sell_quantity = quantity * rebalance_percentage
            
            # Get current price (approximate from portfolio)
            portfolio_summary = portfolio.get_portfolio_summary({})
            if coin not in portfolio_summary.get('holdings', {}):
                continue
            
            current_price = portfolio_summary['holdings'][coin].get('price', 0.0)
            if current_price <= 0:
                continue
            
            # Execute the rebalance sell
            success = portfolio.execute_sell(coin, sell_quantity, current_price)
            
            if success:
                total_value = sell_quantity * current_price
                trade = EnhancedTrade(
                    timestamp=timestamp,
                    coin=coin,
                    action="REBALANCE_SELL",
                    quantity=sell_quantity,
                    price=current_price,
                    total_value=total_value,
                    reason=f"Profit rebalancing - {signal.reason}",
                    tier=0
                )
                rebalance_trades.append(trade)
        
        return rebalance_trades
    
    def _log_trade_to_csv(self, trade: EnhancedTrade) -> None:
        """Log a trade to the CSV file."""
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
                    trade_dict['reason'],
                    trade_dict['tier']
                ])
        except Exception as e:
            print(f"Error logging trade to CSV: {e}")
    
    def get_recent_trades(self, limit: int = 10) -> List[EnhancedTrade]:
        """Get recent trades."""
        return self.trades[-limit:] if len(self.trades) > limit else self.trades
    
    def get_dca_statistics(self) -> Dict[str, Any]:
        """Get DCA-specific trading statistics."""
        if not self.trades:
            return {'total_dca_trades': 0, 'dca_by_tier': {}, 'avg_dca_size': 0.0}
        
        dca_trades = [t for t in self.trades if t.tier > 0]
        dca_by_tier = {}
        
        for trade in dca_trades:
            tier = trade.tier
            if tier not in dca_by_tier:
                dca_by_tier[tier] = {'count': 0, 'total_value': 0.0}
            
            dca_by_tier[tier]['count'] += 1
            dca_by_tier[tier]['total_value'] += trade.total_value
        
        total_dca_value = sum(t.total_value for t in dca_trades)
        avg_dca_size = total_dca_value / len(dca_trades) if dca_trades else 0.0
        
        return {
            'total_dca_trades': len(dca_trades),
            'dca_by_tier': dca_by_tier,
            'avg_dca_size': round_to_precision(avg_dca_size, 2),
            'total_dca_value': round_to_precision(total_dca_value, 2)
        }
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'rebalance_trades': 0,
                'total_volume': 0.0,
                'coins_traded': []
            }
        
        buy_trades = [t for t in self.trades if t.action == 'BUY']
        sell_trades = [t for t in self.trades if t.action in ['SELL', 'REBALANCE_SELL']]
        rebalance_trades = [t for t in self.trades if 'REBALANCE' in t.action]
        
        total_volume = sum(t.total_value for t in self.trades)
        coins_traded = list(set(t.coin for t in self.trades if t.coin != 'PORTFOLIO'))
        
        # Calculate profitability metrics
        total_buys_value = sum(t.total_value for t in buy_trades)
        total_sells_value = sum(t.total_value for t in sell_trades)
        realized_pnl = total_sells_value - total_buys_value
        
        return {
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'rebalance_trades': len(rebalance_trades),
            'total_volume': round_to_precision(total_volume, 2),
            'coins_traded': coins_traded,
            'avg_trade_value': round_to_precision(total_volume / len(self.trades), 2),
            'realized_pnl': round_to_precision(realized_pnl, 2),
            'total_buys_value': round_to_precision(total_buys_value, 2),
            'total_sells_value': round_to_precision(total_sells_value, 2)
        }
    
    def format_trade_summary(self, trades: List[EnhancedTrade]) -> str:
        """Format a list of trades into a readable summary."""
        if not trades:
            return "No trades executed this cycle."
        
        summary_lines = []
        for trade in trades:
            # Enhanced formatting with tier information
            tier_str = f" [T{trade.tier}]" if trade.tier > 0 else ""
            action_symbol = "📈" if trade.action == "BUY" else "📉" if "SELL" in trade.action else "⚖️"
            
            summary_lines.append(
                f"{action_symbol} {trade.action}{tier_str} {trade.quantity:.6f} {trade.coin} "
                f"@ ${trade.price:.2f} = {format_currency(trade.total_value)} "
                f"({trade.reason})"
            )
        
        return "\n".join(summary_lines)
    
    def clear_trade_history(self) -> None:
        """Clear all trade history."""
        self.trades.clear()
        self.rebalance_trades.clear()
        # Recreate the CSV file with just headers
        self._initialize_trade_history_file()