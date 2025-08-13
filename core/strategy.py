"""
Trading strategy implementation using threshold-based buy/sell logic.
"""
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from utils.math_tools import percent_change, calculate_trade_amount


class TradeAction(Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeSignal:
    """Represents a trading signal."""
    
    def __init__(self, coin: str, action: TradeAction, quantity: float = 0.0, 
                 price: float = 0.0, reason: str = ""):
        self.coin = coin
        self.action = action
        self.quantity = quantity
        self.price = price
        self.reason = reason
    
    def __repr__(self) -> str:
        if self.action == TradeAction.HOLD:
            return f"TradeSignal({self.coin}: {self.action.value} - {self.reason})"
        return f"TradeSignal({self.coin}: {self.action.value} {self.quantity:.6f} @ ${self.price:.2f} - {self.reason})"


class ThresholdStrategy:
    """Implements threshold-based trading strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.movement_threshold = config['strategy']['movement_threshold']
        self.trade_amount_percentage = config['strategy']['trade_amount_percentage']
        self.min_trade_amount = config['strategy']['min_trade_amount']
        self.coins = list(config['coins'].keys())
    
    def analyze_market(self, current_prices: Dict[str, float], 
                      portfolio, last_trade_prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Analyze market conditions and generate trade signals.
        
        Args:
            current_prices: Current market prices
            portfolio: Portfolio instance
            last_trade_prices: Last trade prices for each coin
            
        Returns:
            List of trade signals
        """
        signals = []
        
        for coin in self.coins:
            current_price = current_prices.get(coin, 0.0)
            if current_price <= 0:
                continue
            
            signal = self._analyze_coin(coin, current_price, portfolio, last_trade_prices)
            signals.append(signal)
        
        return signals
    
    def _analyze_coin(self, coin: str, current_price: float, portfolio, 
                     last_trade_prices: Dict[str, float]) -> TradeSignal:
        """
        Analyze a specific coin for trading opportunities.
        
        Args:
            coin: Coin symbol
            current_price: Current price
            portfolio: Portfolio instance
            last_trade_prices: Last trade prices
            
        Returns:
            Trade signal for the coin
        """
        last_trade_price = last_trade_prices.get(coin)
        
        # If never traded, use current price as baseline for future trades
        if last_trade_price is None:
            last_trade_prices[coin] = current_price
            movement_needed = self.movement_threshold * 100
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Need {movement_needed:.2f}% more movement"
            )
        
        # Calculate price movement since last trade
        price_change = percent_change(last_trade_price, current_price)
        
        # Check for buy signal (price dropped by threshold or more)
        if price_change <= -self.movement_threshold:
            return self._generate_buy_signal(coin, current_price, portfolio, abs(price_change))
        
        # Check for sell signal (price increased by threshold or more)
        elif price_change >= self.movement_threshold:
            signal = self._generate_sell_signal(coin, current_price, portfolio, price_change)
            # If we can't sell (no holdings), update baseline to current price for future trades
            if signal.action == TradeAction.HOLD and "No holdings to sell" in signal.reason:
                last_trade_prices[coin] = current_price
                return TradeSignal(
                    coin, TradeAction.HOLD, 0.0, current_price,
                    f"Updated baseline to ${current_price:.2f} - Need 3.00% more movement"
                )
            return signal
        
        # No action needed
        else:
            movement_needed = self.movement_threshold - abs(price_change)
            movement_needed_percent = movement_needed * 100
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Need {movement_needed_percent:.2f}% more movement"
            )
    
    def _generate_buy_signal(self, coin: str, current_price: float, 
                           portfolio, price_drop: float) -> TradeSignal:
        """
        Generate a buy signal for a coin.
        
        Args:
            coin: Coin symbol
            current_price: Current price
            portfolio: Portfolio instance
            price_drop: Price drop percentage
            
        Returns:
            Buy trade signal
        """
        usdt_balance = portfolio.get_usdt_balance()
        
        # Calculate trade amount
        trade_amount_usdt = calculate_trade_amount(
            usdt_balance, self.trade_amount_percentage, self.min_trade_amount
        )
        
        if trade_amount_usdt <= 0:
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Insufficient balance for buy (need ${self.min_trade_amount})"
            )
        
        # Calculate quantity to buy
        quantity = trade_amount_usdt / current_price
        
        return TradeSignal(
            coin, TradeAction.BUY, quantity, current_price,
            f"Price dropped {price_drop*100:.2f}%"
        )
    
    def _generate_sell_signal(self, coin: str, current_price: float, 
                            portfolio, price_increase: float) -> TradeSignal:
        """
        Generate a sell signal for a coin.
        
        Args:
            coin: Coin symbol
            current_price: Current price
            portfolio: Portfolio instance
            price_increase: Price increase percentage
            
        Returns:
            Sell trade signal
        """
        holding = portfolio.get_holding(coin)
        
        if holding <= 0:
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"No holdings to sell (price up {price_increase*100:.2f}%)"
            )
        
        # Sell a portion of holdings (same percentage as buy strategy)
        sell_quantity = holding * self.trade_amount_percentage
        sell_value = sell_quantity * current_price
        
        # Check if sell value meets minimum trade amount
        if sell_value < self.min_trade_amount:
            # Sell all holdings if below minimum
            sell_quantity = holding
        
        return TradeSignal(
            coin, TradeAction.SELL, sell_quantity, current_price,
            f"Price increased {price_increase*100:.2f}%"
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy configuration information.
        
        Returns:
            Strategy info dictionary
        """
        return {
            'strategy_type': 'threshold_based',
            'movement_threshold': f"{self.movement_threshold*100:.1f}%",
            'trade_amount_percentage': f"{self.trade_amount_percentage*100:.1f}%",
            'min_trade_amount': f"${self.min_trade_amount:.2f}",
            'tracked_coins': self.coins
        }