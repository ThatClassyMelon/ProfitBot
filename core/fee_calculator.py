"""
Trading fee calculator for Alpaca Markets.
Handles commission calculations and fee-aware position sizing.
"""
from typing import Dict, Any, Tuple
from decimal import Decimal, ROUND_DOWN
from utils.math_tools import round_to_precision


class AlpacaFeeCalculator:
    """Calculate trading fees and commissions for Alpaca Markets."""
    
    def __init__(self):
        """Initialize fee calculator with Alpaca's fee structure."""
        # Alpaca Crypto Trading Fees (as of 2025)
        # Paper trading typically has the same fee structure as live
        
        # Commission rates (percentage of trade value)
        self.crypto_commission_rate = 0.0025  # 0.25% per trade (typical)
        
        # Minimum fees
        self.min_commission = 0.01  # $0.01 minimum per trade
        
        # Spread costs (bid-ask spread) - estimated
        self.estimated_spread_bps = 10  # 10 basis points (0.1%)
        
        # Network fees (blockchain fees) - varies by crypto
        self.network_fees = {
            'BTC/USD': 0.5,   # ~$0.50 per transaction
            'ETH/USD': 1.0,   # ~$1.00 per transaction  
            'SOL/USD': 0.01,  # ~$0.01 per transaction
            'XRP/USD': 0.01,  # ~$0.01 per transaction
        }
    
    def calculate_trading_fees(self, symbol: str, trade_value: float, 
                             trade_type: str = 'market') -> Dict[str, float]:
        """
        Calculate all fees for a trade.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            trade_value: Dollar value of the trade
            trade_type: Type of order ('market' or 'limit')
            
        Returns:
            Dictionary with fee breakdown
        """
        # Commission fee
        commission = max(
            trade_value * self.crypto_commission_rate,
            self.min_commission
        )
        
        # Spread cost (only for market orders)
        spread_cost = 0.0
        if trade_type == 'market':
            spread_cost = trade_value * (self.estimated_spread_bps / 10000)
        
        # Network fee (blockchain fee)
        network_fee = self.network_fees.get(symbol, 0.1)
        
        # Total fees
        total_fees = commission + spread_cost + network_fee
        
        return {
            'commission': round_to_precision(commission, 4),
            'spread_cost': round_to_precision(spread_cost, 4),
            'network_fee': round_to_precision(network_fee, 4),
            'total_fees': round_to_precision(total_fees, 4),
            'fee_percentage': round_to_precision((total_fees / trade_value) * 100, 3)
        }
    
    def calculate_round_trip_fees(self, symbol: str, trade_value: float) -> Dict[str, float]:
        """
        Calculate fees for a complete round trip (buy + sell).
        
        Args:
            symbol: Trading symbol
            trade_value: Dollar value of the position
            
        Returns:
            Dictionary with round-trip fee breakdown
        """
        buy_fees = self.calculate_trading_fees(symbol, trade_value, 'market')
        sell_fees = self.calculate_trading_fees(symbol, trade_value, 'market')
        
        total_round_trip = buy_fees['total_fees'] + sell_fees['total_fees']
        
        return {
            'buy_fees': buy_fees['total_fees'],
            'sell_fees': sell_fees['total_fees'],
            'total_round_trip': round_to_precision(total_round_trip, 4),
            'round_trip_percentage': round_to_precision((total_round_trip / trade_value) * 100, 3)
        }
    
    def adjust_position_size_for_fees(self, symbol: str, available_cash: float, 
                                    target_percentage: float = 0.1) -> Tuple[float, Dict[str, float]]:
        """
        Calculate optimal position size accounting for fees.
        
        Args:
            symbol: Trading symbol
            available_cash: Available cash for trading
            target_percentage: Target percentage of cash to use
            
        Returns:
            Tuple of (adjusted_position_size, fee_info)
        """
        # Calculate base position size
        base_position = available_cash * target_percentage
        
        # Calculate fees for this position
        round_trip_fees = self.calculate_round_trip_fees(symbol, base_position)
        total_fees = round_trip_fees['total_round_trip']
        
        # Adjust position to ensure we have enough cash for fees
        # Reserve extra cash for fees (1.5x estimated fees for safety)
        fee_buffer = total_fees * 1.5
        adjusted_position = base_position - fee_buffer
        
        # Ensure position is not too small
        min_position = 10.0  # $10 minimum
        if adjusted_position < min_position:
            adjusted_position = 0.0  # Skip trade if too small
        
        fee_info = {
            'base_position': round_to_precision(base_position, 2),
            'adjusted_position': round_to_precision(adjusted_position, 2),
            'estimated_fees': round_to_precision(total_fees, 4),
            'fee_buffer': round_to_precision(fee_buffer, 4),
            'fee_percentage': round_trip_fees['round_trip_percentage']
        }
        
        return adjusted_position, fee_info
    
    def calculate_breakeven_price(self, entry_price: float, symbol: str, 
                                position_value: float) -> float:
        """
        Calculate the price needed to break even after fees.
        
        Args:
            entry_price: Price at which position was entered
            symbol: Trading symbol
            position_value: Total position value
            
        Returns:
            Breakeven price including all fees
        """
        round_trip_fees = self.calculate_round_trip_fees(symbol, position_value)
        total_fees = round_trip_fees['total_round_trip']
        
        # Calculate quantity from position value and entry price
        quantity = position_value / entry_price
        
        # Breakeven price = entry_price + (total_fees / quantity)
        breakeven_price = entry_price + (total_fees / quantity)
        
        return round_to_precision(breakeven_price, 6)
    
    def is_trade_profitable_after_fees(self, entry_price: float, current_price: float,
                                     symbol: str, position_value: float, 
                                     action: str = 'sell') -> Dict[str, Any]:
        """
        Check if a trade would be profitable after accounting for fees.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            symbol: Trading symbol
            position_value: Total position value
            action: 'sell' for long positions
            
        Returns:
            Dictionary with profitability analysis
        """
        # Calculate fees for closing the position
        closing_fees = self.calculate_trading_fees(symbol, position_value, 'market')
        
        # Calculate gross profit/loss
        quantity = position_value / entry_price
        current_value = quantity * current_price
        gross_pnl = current_value - position_value
        
        # Calculate net profit/loss after fees
        net_pnl = gross_pnl - closing_fees['total_fees']
        
        # Calculate breakeven price
        breakeven_price = self.calculate_breakeven_price(entry_price, symbol, position_value)
        
        return {
            'is_profitable': net_pnl > 0,
            'gross_pnl': round_to_precision(gross_pnl, 4),
            'net_pnl': round_to_precision(net_pnl, 4),
            'closing_fees': closing_fees['total_fees'],
            'breakeven_price': breakeven_price,
            'current_price': current_price,
            'profit_margin': round_to_precision((net_pnl / position_value) * 100, 3)
        }